import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import argparse
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument(
    '--task',
    type=str,
    help='which task to run. Choose from (pred_single|pred_transfer|pred_single_baseline|pred_transfer_baseline|gen_baseline)',
    choices=['pred_single', 'pred_transfer', 'pred_single_baseline', 'pred_transfer_baseline', 'gen_baseline']
)
parser.add_argument(
    '--dataset',
    type=str,
    help='Datasets. Choose from (synthetic|compas|law_school|adult|ACSIncome)',
    choices=['synthetic', 'compas', 'law_school', 'adult', 'ACSIncome']
)
parser.add_argument(
    '--use_gpu',
    type=bool,
    choices=[True, False]
)
parser.add_argument(
    '--gpu_idx',
    type=str,
    default='0',
    help='CUDA device index'
)
args = parser.parse_args()

CONF_PATH = {
    'synthetic_single': 'conf/cfg_synthetic_single.yaml',
    'synthetic_transfer': 'conf/cfg_synthetic_transfer.yaml',
    'compas': 'conf/cfg_compas.yaml',
    'law_school': 'conf/cfg_law_school.yaml',
    'adult': 'conf/cfg_adult.yaml',
    'ACSIncome': 'conf/cfg_ACSIncome_transfer.yaml'
}

VALID_DATASETS = {
    'pred_single': ['synthetic', 'compas', 'law_school', 'adult'],
    'pred_single_baseline': ['synthetic', 'compas', 'law_school', 'adult'],
    'pred_transfer': ['synthetic', 'ACSIncome'],
    'pred_transfer_baseline': ['synthetic', 'ACSIncome'],
    'gen_baseline': ['synthetic']
}

def validate_dataset(args):
    if args.dataset not in VALID_DATASETS.get(args.task, []):
        raise ValueError(f"Invalid dataset '{args.dataset}' for task '{args.task}'.
                         Valid options are: {VALID_DATASETS[args.task]}")


if __name__ == '__main__':
    # get config
    dat = args.dataset
    if args.task == 'pred_single':
        cfg_file = CONF_PATH[dat+'_single'] if dat == 'synthetic' else CONF_PATH[dat]
    elif args.task == 'pred_transfer':
        cfg_file = CONF_PATH[dat]
        
    cfg = OmegaConf.load(cfg_file)
    cfg.use_gpu = args.use_gpu
    if args.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_idx
    
    if args.task == 'pred_single':
        assert args.dataset in ['synthetic', 'compas', 'law_school', 'adult'], "Wrong dataset"
        from exp.main_single import main
        main(cfg_file)
    elif args.task == 'pred_transfer':
        