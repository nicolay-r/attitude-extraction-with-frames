from os import path
from os.path import dirname, join
from core.common.utils import create_dir_if_not_exists


def get_data_root():
    return path.join(dirname(__file__), u"data/")


def get_eval_root():
    return path.join(get_data_root(), u"Eval/")


def get_log_root():
    log_root_dir = path.join(get_data_root(), u"Log/")
    create_dir_if_not_exists(log_root_dir)
    return log_root_dir


EXPERIMENTS_NAME = u'rusentrel'


def get_experiments_dir():
    target_dir = join(get_data_root(), u"./{}/".format(EXPERIMENTS_NAME))
    create_dir_if_not_exists(target_dir)
    return target_dir