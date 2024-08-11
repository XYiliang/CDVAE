# coding = utf-8
from datasets.compas.compas_loaders import make_compas_loaders
from datasets.law_school.law_loaders import make_law_loaders
from datasets.adult.adult_loaders import make_adult_loaders
from datasets.ACSIncome.ACSIncome_loaders import make_ACSIncome_loaders
from datasets.synthetic.synthetic_loaders import make_synthetic_loaders


def get_loaders(cfg):
    loaders = {
        'synthetic': make_synthetic_loaders,
        'compas': make_compas_loaders,
        'law_school': make_law_loaders,
        'adult': make_adult_loaders,
        'ACSIncome': make_ACSIncome_loaders
    }
    return loaders[cfg.dataset_name](cfg)
