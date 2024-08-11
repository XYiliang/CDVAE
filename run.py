import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import argparse
from omegaconf import OmegaConf

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
    'pred_transfer_baseline': ['synthetic', 'ACSIncome']
}

def validate_args(args):
    if args.task not in VALID_DATASETS:
        raise ValueError(f"Invalid task. Valid options are: {list(VALID_DATASETS.keys())}")
    if args.dataset not in VALID_DATASETS.get(args.task, []):
        raise ValueError(f"Invalid dataset {args.dataset} for task {args.task}. Valid options are: {VALID_DATASETS[args.task]}")


if __name__ == '__main__':
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
        '--generative_model',
        type=str,
        help='Datasets. Choose from (CDVAE|CEVAE|mCEVAE|DCEVAE)',
        choices=["CDVAE", "CEVAE", "mCEVAE", "DCEVAE"]
    )
    parser.add_argument(
        '--use_gpu',
        type=bool,
        default='false'
    )
    parser.add_argument(
        '--gpu_ids',
        type=str,
        default='0',
        help='GPU index'
    )
    args = parser.parse_args()

    # get config
    dat = args.dataset
    if args.task == 'pred_single' or args.task == 'pred_single_baseline':
        cfg_file = CONF_PATH[dat+'_single'] if dat == 'synthetic' else CONF_PATH[dat]
    elif args.task == 'pred_transfer' or args.task == 'pred_transfer_baseline':
        cfg_file = CONF_PATH[dat+'_transfer'] if dat == 'synthetic' else CONF_PATH[dat]
    
    cfg = OmegaConf.load(cfg_file)
    cfg.use_gpu = args.use_gpu
    
    if args.use_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    
    if args.task == 'pred_single':
        from exp.main_single import main
    elif args.task == 'pred_transfer':
        from exp.main_transfer import main
    elif args.task == 'pred_single_baseline':
        from exp.predictive_baselines_single import main
    elif args.task == 'pred_transfer_baseline':
        from exp.predictive_baselines_transfer import main
    
    main(cfg)
