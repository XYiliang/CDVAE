import os
import sys

# print("系统路径", sys.path)
sys.path += [r'F:\\PythonProjects\\VAE\\robust_fair_covariate_shift', r'F:\\PythonProjects\\VAE', r'F:\\Python\\python39.zip', r'F:\\Python\\DLLs', r'F:\\Python\\lib', r'F:\\Python', r'F:\\PythonProjects\\VAE\\venv', r'F:\\PythonProjects\\VAE\\venv\\lib\\site-packages', r'C:\\Users\\夏亦良\\AppData\\Roaming\\Python\\Python39\\site-packages', 'F:\\Python\\lib\\site-packages', 'F:\\Python\\lib\\site-packages\\pip-23.3.2-py3.9.egg', r'F:\\Python\\lib\\site-packages\\win32', r'F:\\Python\\lib\\site-packages\\win32\\lib', r'F:\\Python\\lib\\site-packages\\Pythonwin']

os.chdir('F:\PythonProjects\VAE')
import dill
import numpy as np
import pandas as pd
import argparse
import logging

from robust_fair_covariate_shift.fair_covariate_shift import eopp_fair_covariate_shift_logloss
from robust_fair_covariate_shift.create_shift import create_shift, process_data
from robust_fair_covariate_shift import prepare_data

data2prepare = {
    "compas": prepare_data.prepare_compas,
    "german": prepare_data.prepare_german,
    "drug": prepare_data.prepare_drug,
    "arrhythmia": prepare_data.prepare_arrhythmia,
    "ACSIncome": prepare_data.prepare_new_adult,
    "synthetic": prepare_data.prepare_synthetic,
}

dataset2reg = {
    "compas": 0.001,
    "german": 0.01,
    "drug": 0.001,
    "arrhythmia": 0.01,
    "ACSIncome": 0.01,
    "synthetic": 0.01
}

dataset2eps = {
    "compas": 0.001,
    "german": 0.001,
    "drug": 0.001,
    "arrhythmia": 0.001,
    "ACSIncome": 0.001,
    "synthetic": 0.001
}

sample_size_ratio = 0.4


def load_dataset(dataset, alpha, beta, kdebw, epsilon, logger):
    src_dataA, src_dataY, src_dataX, trg_dataA, trg_dataY, trg_dataX, trg_test_ratio_idx = data2prepare[dataset]()
    tr_X, tr_A, tr_Y, ts_X, ts_A, ts_Y, tr_ratio, ts_ratio = process_data(src_dataX, src_dataA, src_dataY,
                                                                          trg_dataX, trg_dataA, trg_dataY,
                                                                          args.dataset,
                                                                          kdebw=kdebw, eps=epsilon,
                                                                          logger=logger)

    # trg_test_ratio = ts_ratio[trg_test_ratio_idx[0]:trg_test_ratio_idx[1]]
    # np.savetxt(rf'F:/PythonProjects/VAE/datasets/ACSIncome/50_states_testdata/{dataset}_trg_test_ratio.csv',
    #            trg_test_ratio)

    np.savetxt(r"F:\PythonProjects\VAE\datasets\ACSIncome\50_states_testdata\50_states_train_ratio.csv", tr_ratio)
    np.savetxt(r"F:\PythonProjects\VAE\datasets\ACSIncome\50_states_testdata\50_states_test_ratio.csv", ts_ratio)

    logger.info('ratio saved')

    dataset = dict(
        X_src=tr_X,
        A_src=tr_A,
        Y_src=tr_Y,
        ratio_src=tr_ratio,
        X_trg=ts_X,
        A_trg=ts_A,
        Y_trg=ts_Y,
        ratio_trg=ts_ratio,
    )

    return dataset


if __name__ == "__main__":
    print('start')
    log_dir=r"F:\PythonProjects\VAE\datasets\ACSIncome\50_states_testdata"
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    """保存日志"""
    file_handler = logging.FileHandler(os.path.join(log_dir, '50_states.log'))
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
     
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default='compas',
        type=str,
        required=True,
        help='Dataset name : ["compas","german","drug","arrhythmia", "ACSIncome", "synthetic"].',
    )
    parser.add_argument(
        "--repeat",
        type=int,
        required=False,
        default=1,
        help="number of random shuffle runs.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        required=False,
        default=1,
        help="Shift the Gaussian mean -> mean + alpha in sampling of covariates.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        required=False,
        default=2,
        help="Scale the Gaussian std -> std / beta in sampling of covariates.",
    )
    parser.add_argument(
        "--mu_range",
        type=float,
        required=False,
        nargs="+",
        # 这里记得改回[-1.5, 1.5]
        default=[-1.5, 1.5],
        # default=[-0.5, 0.5],
        help="The search range for \mu - the fairness penalty weight.",
    )

    args = parser.parse_args()
    n = args.repeat
    dataset = args.dataset
    alpha = args.alpha
    beta = args.beta
    C = dataset2reg[dataset]
    eps = dataset2eps[dataset]
    kdebw = 0.3  # KDE bandwidth
    mu_range = args.mu_range
    errs, violations = [], []

    for i in range(n):
        logger.info(
            "------------------------------- {} sample {:d} / {:d}, shift parameters: alpha = {}, beta = {}---------------------------------".format(
                dataset, i + 1, n, alpha, beta
            )
        )

        sample = load_dataset(dataset, alpha, beta, kdebw=kdebw, epsilon=eps, logger=logger)

        h = eopp_fair_covariate_shift_logloss(
            verbose=1, tol=1e-7, random_initialization=False
        )
        h.trg_grp_marginal_matching = True
        h.C = C
        h.max_epoch = 3
        h.max_iter = 3000
        h.tol = 1e-7
        h.random_start = True
        h.verbose = 1
        logger.info('start_fitting')
        h.fit(
            sample["X_src"],
            sample["Y_src"],
            sample["A_src"],
            sample["ratio_src"],
            sample["X_trg"],
            sample["A_trg"],
            sample["ratio_trg"],
            mu_range=mu_range,
        )
        err = 1 - h.score(
            sample["X_trg"], sample["Y_trg"], sample["A_trg"], sample["ratio_trg"]
        )
        violation = abs(
            h.fairness_violation(
                sample["X_trg"], sample["Y_trg"], sample["A_trg"], sample["ratio_trg"]
            )
        )
        errs.append(err)
        violations.append(violation)
        logger.info(
            "Test  - prediction_err : {:.3f}\t fairness_violation : {:.3f} ".format(
                err, violation
            )
        )
        logger.info("Mu = {:.4f}".format(h.mu))
        logger.info("")

    save_dir = r'F:\PythonProjects\VAE\datasets\ACSIncome\50_states_testdata'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    with open(os.path.join(save_dir, f'50_states_{args.dataset}.pkl'), 'wb') as f:
        dill.dump(h, f)

    logger.info(
        "------------------------------- Summary: {}, {:d} samples, shift parameters: alpha = {}, beta = {}---------------------------------".format(
            dataset, n, alpha, beta
        )
    )
    errs = np.array(errs, dtype=float)
    violations = np.array(violations, dtype=float)
    logger.info(
        "Test  - prediction_err : {:.3f} \u00B1 {:.3f} \t fairness_violation : {:.3f} \u00B1 {:.3f} ".format(
            errs.mean(),
            1.96 / np.sqrt(n) * errs.std(),
            violations.mean(),
            1.96 / np.sqrt(n) * violations.std(),
        )
    )
