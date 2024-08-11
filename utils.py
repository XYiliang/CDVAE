# coding = utf-8
import os
import re
import logging
import random
import datetime
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import joblib

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from sklearn.manifold import TSNE
from functools import wraps

def tensor2numpy(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        args = [arg.numpy() if isinstance(arg, torch.Tensor) else arg for arg in args]
        kwargs = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapper

def numpy2tensor(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        args = [torch.from_numpy(arg) if isinstance(arg, np.ndarray) else arg for arg in args]
        kwargs = {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapper

def sync_device(device, x):
    return x.to(device)


def ini_weight(modules):
    for _ in modules:
        if isinstance(_, nn.Linear) or isinstance(_, nn.Conv2d) or isinstance(_, nn.ConvTranspose2d):
            nn.init.xavier_normal_(_.weight.data)


def ini_model_dir(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)


def seed_everything(cfg):
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if cfg.use_gpu and torch.cuda.is_available():
            print("Current CUDA random seed", torch.cuda.initial_seed())
        else:
            print("Current CPU random seed", torch.initial_seed())
    else:
        print('No random seed.')


def save_vae(model, vae_name, cfg):
    device = next(model.parameters()).device
    model.cpu()
    if cfg.is_transfer:
        model_save_dir = f'result/{cfg.dataset_name}/vae_transfer'
    else:
        model_save_dir = f'result/{cfg.dataset_name}/vae_single'
    if not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)
    model_save_path = os.path.join(model_save_dir, vae_name)
    model_save_path = get_new_filepath(model_save_path)
    torch.save(model.state_dict(), model_save_path)
    model.to(device)


def save_predictors(models, vae_savename, cfg):
    for name, model in models.items():

        if cfg.is_transfer:
            model_save_dir = f'result/{cfg.dataset_name}/clf_transfer'
        else:
            model_save_dir = f'result/{cfg.dataset_name}/clf_single'
        os.makedirs(model_save_dir, exist_ok=True)
        
        predictor_savename = re.sub(r'vae_.*\(', f'{name}(', vae_savename.replace(').pth', '')) +\
            f'_hlayers{cfg.mlp_hidden_layers}).pth'
        model_save_path = os.path.join(model_save_dir, predictor_savename)
        model_save_path = get_new_filepath(model_save_path)
        if isinstance(model, nn.Module):
            torch.save(model, model_save_path)
        else:
            joblib.dump(model, model_save_path)


def ini_result_dataframe(dataset_name, task, is_transfer=False):
    model = ['MLP']   # You can add other predictive models by modifing this list and lists in train_test.py
    domains = ['Source', 'Target']

    if task == 'classification':
        metrics = ['Accuracy', 'AUC', '|TE|', 'CFD', 'MMD_rf', 'MMD_rc', 'MMD_cf']
    else:
        metrics = ['RMSE', 'RMSE_cf', 'CFD', 'MMD_rf', 'MMD_rc', 'MMD_cf']

    if is_transfer:
        multi_index = pd.MultiIndex.from_product([domains, metrics], names=['domain', 'metrics'])
        multi_index = multi_index.insert(loc=0, item=('MMD', 'mmd_srcU-trgU'))
    else:
        if dataset_name == 'synthetic':
            metrics += ['real_|TE|', 'real_CFD']
        multi_index = pd.MultiIndex.from_product([model, metrics], names=['model', 'metrics'])

    return pd.DataFrame(index=multi_index)


def ini_baseline_res_df(dataset_name, task, is_transfer=False):
    model = ['MLP']   # You can add other predictive models by modifing this list and lists in train_test.py
    domains = ['Target']

    if task == 'classification':
        metrics = ['Accuracy', 'AUC', '|TE|', 'CFD']
    else:
        metrics = ['RMSE', 'RMSE_cf', 'CFD']

    if is_transfer:
        multi_index = pd.MultiIndex.from_product([domains, metrics], names=['domain', 'metrics'])
    else:
        if dataset_name == 'synthetic':
            metrics += ['real_|TE|', 'real_CFD']
        multi_index = pd.MultiIndex.from_product([model, metrics], names=['model', 'metrics'])

    return pd.DataFrame(index=multi_index)

def get_result_dataframe(cfg):
    if not os.path.isdir('result'):
        os.mkdir('result')
        
    assert cfg.dataset_name in ('synthetic', 'compas', 'law_school', 'adult', 'ACSIncome'), \
        'Wrong dataset name, check cfg.dataset_name.'
    
    result_dir = os.path.join('result', cfg.dataset_name)
    if not os.path.isdir(result_dir):
        os.mkdir(result_dir)
    
    suffix = 'transfer' if cfg.is_transfer else 'single'
    csv_path = os.path.join(result_dir, f'res_{cfg.dataset_name}_{suffix}.csv')
    
    if os.path.isfile(csv_path):
        return pd.read_csv(csv_path, index_col=[0, 1])
    else:
        df = ini_result_dataframe(cfg.dataset_name, cfg.task, cfg.is_transfer)
        return df
        

def save_result_dataframe(res_df, cfg):
    result_dir = os.path.join('result', cfg.dataset_name)
    suffix = 'transfer' if cfg.is_transfer else 'single'
    csv_path = os.path.join(result_dir, f'res_{cfg.dataset_name}_{suffix}.csv')
    res_df.to_csv(csv_path)


def get_writer(log_dir, vae_savename, dataset_name):
    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)
    time = datetime.datetime.now().strftime("%Y%m%d%H%M")
    writer_name = dataset_name + time + vae_savename
    path = os.path.join(log_dir, writer_name)
    writer = SummaryWriter(log_dir=path)

    return writer


def get_logger(log_dir):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    file_handler = logging.FileHandler(os.path.join(log_dir, 'exp.log'), encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger, console_handler, file_handler


def filter_dummy_x(xnc, xnb, xac, xab, dim_dict):
    xn_list = []
    xa_list = []
    if dim_dict['xnc'] != 0: xn_list.append(xnc)
    if dim_dict['xnb'] != 0: xn_list.append(xnb)
    if dim_dict['xac'] != 0: xa_list.append(xac)
    if dim_dict['xab'] != 0: xa_list.append(xab)

    xn = torch.cat(xn_list, 1)
    xa = torch.cat(xa_list, 1)

    return xn, xa


def get_dims(dim_dict):
    xnc_dim = dim_dict['xnc']
    xnb_dim = dim_dict['xnb']
    xac_dim = dim_dict['xac']
    xab_dim = dim_dict['xab']
    a_dim = dim_dict['a']
    y_dim = dim_dict['y']

    return xnc_dim, xnb_dim, xac_dim, xab_dim, a_dim, y_dim


def onehot(x: np.ndarray, num_classes) -> np.ndarray:
    return np.eye(num_classes)[x].squeeze().astype(float)


def binning(x, bins):
    return np.digitize(x, bins, right=True)


def get_new_filepath(filepath):
    counter = 0
    dirname, filename = os.path.split(filepath)

    while os.path.exists(os.path.join(dirname, f'{counter}_' + filename)):
        counter += 1

    filepath = os.path.join(dirname, f'{counter}_' + filename)
    return filepath


def get_new_colname(col_name, df):
    counter = 0
    while f'{counter}_' + col_name in df.columns:
        counter += 1
    return f'{counter}_' + col_name


def draw_tSNE(u1, u2, cfg, save_name, save_fig=True, comparison='do', part_size=500, mmd_val=None, ylim=None):
    """
    :param u1: latent representation 1
    :param u2: latent representation 2
    :param save_fig: if True, save fig
    :param comparison: ['do', 'distill', 'src-trg']
    :param part_size: How many dots to plot
    """
    if not isinstance(u1, np.ndarray):
        u1, u2 = u1.detach().cpu().numpy(), u2.detach().cpu().numpy()

    if cfg.is_transfer:
        title = r'$\lambda_{CSS}=' + f'{cfg.parm_css}$' + r'$\ \ \lambda_{TF}=' + f'{cfg.parm_tf}$'
    else:
        title = r'$\lambda_{CSS}=' + f'{cfg.parm_css}$'

    save_dir = {
        'do': f'result/{cfg.dataset_name}/t-SNE/tSNE una-cf_una',
        'distill': f'result/{cfg.dataset_name}/t-SNE/tSNE una-cf_una_distilled',
        'src-trg': f'result/{cfg.dataset_name}/t-SNE/tSNE src_una-trg_una'
    }
    label = {
        'do': (r'$U_{A \leftarrow a}$', r"$U_{A \leftarrow a^{\prime}}$"),
        'distill': (r'$U_{A \leftarrow a}$', r"$Distilled \ U_{A \leftarrow a^{\prime}}$"),
        'src-trg': (r'$U_{Source}$', r'$U_{Target}$'),
    }

    tsne = TSNE(n_components=2, random_state=cfg.seed)
    u1, u2 = u1[:part_size], u2[:part_size]
    latent_tsne = tsne.fit_transform(np.concatenate([u1, u2], axis=0))
    factual_latent_dots = [latent_tsne[:u1.shape[0], 0], latent_tsne[:u1.shape[0], 1]]
    counter_latent_dots = [latent_tsne[u1.shape[0]:, 0], latent_tsne[u1.shape[0]:, 1]]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(factual_latent_dots[0], factual_latent_dots[1], c='C0', label=label[comparison][0], marker='o')
    ax.scatter(counter_latent_dots[0], factual_latent_dots[1], c='C1', label=label[comparison][1], marker='x')
    ax.set_title(title, fontsize=16)
    ax.legend(loc='upper right', fontsize=16)
    if comparison == 'src-trg':
        fig.text(0, 1, f'MMD={mmd_val:.3e}', va='top', ha='left')
        ax.text(0.01, 0.99, f'MMD={mmd_val:.4e}', transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left',
                fontsize=10, color='blue', weight='bold')
    ax.set_ylim(ylim)
    
    if save_fig:
        try:
            if not os.path.isdir(f'result/{cfg.dataset_name}/t-SNE/'):
                os.makedirs(f'result/{cfg.dataset_name}/t-SNE/', exist_ok=True)
            if not os.path.isdir(save_dir[comparison]):
                os.mkdir(save_dir[comparison])
            save_path = os.path.join(save_dir[comparison], save_name)
            if os.path.isfile(save_path):
                save_path = get_new_filepath(save_path)
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0.1, format='tif', dpi=900)
            print('t-SNE fig saved')
        except Exception as e:
            print(e)
            Warning(f'Due to the exception above, the fig will be save to result/{cfg.dataset_name}/temp_tSNE')
            if not os.path.isdir(f'result/{cfg.dataset_name}/temp_tSNE'):
                os.mkdir(f'result/{cfg.dataset_name}/t-SNE/temp_tSNE')
            fig.savefig(f'result/{cfg.dataset_name}/t-SNE/temp_tSNE/{save_name}', bbox_inches='tight', pad_inches=0.1,
                        format='tif', dpi=900)
            print('t-SNE fig saved to temp_tSNE')


def summary(df):
    summary_list = []
    for idx, row in df.iterrows():
        max_val, min_val = np.max(row), np.min(row)
        mid, diff = round((max_val + min_val) / 2, 4), round((max_val - min_val) / 2, 4)
        summary_list.append(f'{mid}±{diff}')
    return summary_list
