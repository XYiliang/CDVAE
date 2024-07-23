import sys
import dill
import numpy as np
from scipy.stats import norm
from sklearn.decomposition import PCA
from KDEpy import NaiveKDE
from timeit import default_timer

trg_dir = r'F:\PythonProjects\VAE\datasets\ACSIncome\50_states_testdata'

def process_data(source_data, source_a, source_y, target_data, target_a, target_y, dataset, logger, kdebw=0.3, eps=0.001):
    source_data, source_a, source_y, target_data, target_a, target_y\
        = over_sample(source_data, source_a, source_y, target_data, target_a, target_y)

    # start = default_timer()
    # pca = PCA(n_components=2)
    # data = np.vstack([source_data, target_data])
    # pc2 = pca.fit_transform(data)
    # src_kde = KDEAdapter(kde=NaiveKDE(kernel="gaussian", bw=kdebw)).fit(pc2[:len(source_data), :])
    # trg_kde = KDEAdapter(kde=NaiveKDE(kernel="gaussian", bw=kdebw)).fit(pc2[len(source_data):, :])
    # with open(f'trg_dir/{dataset}_src_kde.pkl', 'wb') as f:
    #     dill.dump(src_kde, f)
    # with open(f'{trg_dir}/{dataset}_trg_kde.pkl', 'wb') as f:
    #     dill.dump(trg_kde, f)
    # with open(f'robust_fair_covariate_shift/{dataset}_src_kde.pkl', 'rb') as f:
    #     src_kde = dill.load(f)
    # with open(f'datasets/ACSIncome/50_states_testdata/new_adult_trg_kde.pkl', 'rb') as f:
    #     trg_kde = dill.load(f)
    # logger.info('kde start evaluating')
    # ratios = src_kde.p(pc2, eps) / trg_kde.p(pc2, eps)
    # src_ratios, trg_ratios = ratios[:len(source_data)], ratios[len(source_data):]
    # end = default_timer()
    # logger.info('KDE consume time :', round((end-start)/60, 2), 'mins')

    src_ratios = np.loadtxt(r"F:\PythonProjects\VAE\datasets\ACSIncome\50_states_testdata\50_states_train_ratio.csv")
    trg_ratios = np.loadtxt(r"F:\PythonProjects\VAE\datasets\ACSIncome\50_states_testdata\50_states_test_ratio.csv")

    return source_data, source_a, source_y.squeeze(), target_data, target_a, target_y.squeeze(), src_ratios, trg_ratios


def estimate_ratio(pc2, src_sample_size, eps, dataset_name):
    if dataset_name == 'ACSIncome':
        with open('robust_fair_covariate_shift/new_adult_src_kde.pkl', 'rb') as f:
            src_kde = dill.load(f)
        with open('robust_fair_covariate_shift/new_adult_trg_kde.pkl', 'rb') as f:
            trg_kde = dill.load(f)
    else:
        with open('robust_fair_covariate_shift/synthetic_src_kde.pkl', 'rb') as f:
            src_kde = dill.load(f)
        with open('robust_fair_covariate_shift/synthetic_trg_kde.pkl', 'rb') as f:
            trg_kde = dill.load(f)
    ratios = src_kde.p(pc2, eps) / trg_kde.p(pc2, eps)
    src_ratios, trg_ratios = ratios[:src_sample_size], ratios[src_sample_size:]
    return src_ratios, trg_ratios


def over_sample(source_data, source_a, source_y, target_data, target_a, target_y):
    src_size, trg_size = source_data.shape[0], target_data.shape[0]
    if src_size > trg_size:
        idx = np.random.choice(range(trg_size), size=src_size-trg_size, replace=True)
        target_data = np.vstack([target_data, target_data[idx]])
        target_a = np.hstack([target_a, target_a[idx]])
        target_y = np.hstack([target_y, target_y[idx]])
    elif src_size < trg_size:
        idx = np.random.choice(range(src_size), size=trg_size-src_size, replace=True)
        source_data = np.vstack([source_data, source_data[idx]])
        source_a = np.hstack([source_a, source_a[idx]])
        source_y = np.hstack([source_y, source_y[idx]])

    return source_data, source_a, source_y, target_data, target_a, target_y


def create_shift(
    data,
    src_split=0.4,
    alpha=1,
    beta=2,
    kdebw=0.3,
    eps=0.001,
):
    """
    Creates covariate shift sampling of data into disjoint source and target set.

    Let \mu and \sigma be the mean and the standard deviation of the first principal component retrieved by PCA on the whole data.
    The target is randomly sampled based on a Gaussian with mean = \mu and standard deviation = \sigma.
    The source is randomly sampled based on a Gaussian with mean = \mu + alpha and standard devaition = \sigma / beta

    data: [m, n]
    alpha, beta: the parameter that distorts the gaussian used in sampling
                   according to the first principle component
    output: source indices, target indices, ratios based on kernel density estimation with bandwidth = kdebw and smoothed by eps
    """
    m = np.shape(data)[0]
    source_size = int(m * src_split)
    target_size = source_size

    pca = PCA(n_components=2)
    pc2 = pca.fit_transform(data)
    pc = pc2[:, 0]
    pc = pc.reshape(-1, 1)

    pc_mean = np.mean(pc)
    pc_std = np.std(pc)

    sample_mean = pc_mean + alpha
    sample_std = pc_std / beta

    # sample according to the probs
    prob_s = norm.pdf(pc, loc=sample_mean, scale=sample_std)
    sum_s = np.sum(prob_s)
    prob_s = prob_s / sum_s
    prob_t = norm.pdf(pc, loc=pc_mean, scale=pc_std)
    sum_t = np.sum(prob_t)
    prob_t = prob_t / sum_t

    source_ind = np.random.choice(
        range(m), size=source_size, replace=False, p=np.reshape(prob_s, (m))
    )

    pt_proxy = np.copy(prob_t)
    pt_proxy[source_ind] = 0
    pt_proxy = pt_proxy / np.sum(pt_proxy)
    target_ind = np.random.choice(
        range(m), size=target_size, replace=False, p=np.reshape(pt_proxy, (m))
    )

    assert np.all(np.sort(source_ind) != np.sort(target_ind))

    src_kde = KDEAdapter(kde=NaiveKDE(kernel="gaussian", bw=kdebw)).fit(
        pc2[source_ind, :]
    )
    trg_kde = KDEAdapter(kde=NaiveKDE(kernel="gaussian", bw=kdebw)).fit(
        pc2[target_ind, :]
    )

    ratios = src_kde.p(pc2, eps) / trg_kde.p(pc2, eps)
    print("min ratio= {:.5f}, max ratio= {:.5f}".format(np.min(ratios), np.max(ratios)))

    return source_ind, target_ind, ratios


class KDEAdapter:
    def __init__(self, kde=NaiveKDE(kernel="gaussian", bw=0.3)):
        self._kde = kde

    def fit(self, sample):
        self._kde.fit(sample)
        return self

    def pdf(self, sample):
        density = self._kde.evaluate(sample)
        return density

    def p(self, sample, eps=0.0):
        density = self._kde.evaluate(sample)
        return (density + eps) / np.sum(density + eps)
