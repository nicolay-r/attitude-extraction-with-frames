#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys


sys.path.append('../../../')

from core.networks.context.configurations.cnn import CNNConfig
from core.networks.context.architectures.cnn import VanillaCNN
from core.evaluation.evaluators.two_class import TwoClassEvaluator

from rusentrel.ctx_names import ModelNames
from rusentrel.utils import run_testing
from rusentrel.callback import CustomCallback
from rusentrel.rusentrel_ds.common import DS_NAME_PREFIX, \
    ds_ctx_common_config_settings, \
    ds_common_callback_modification_func

from experiments.rusentrel_ds.rusentrel_ds_io import RuSentRelWithRuAttitudesIO
from experiments.rusentrel.context.model import ContextLevelTensorflowModel
from rusentrel.classic.ctx.cnn import ctx_cnn_custom_config


def run_testing_cnn(
        name_prefix=DS_NAME_PREFIX,
        cv_count=1,
        common_callback_func=ds_common_callback_modification_func):

    run_testing(model_name=name_prefix + ModelNames.CNN,
                create_network=VanillaCNN,
                create_config=CNNConfig,
                create_io=RuSentRelWithRuAttitudesIO,
                create_model=ContextLevelTensorflowModel,
                evaluator_class=TwoClassEvaluator,
                create_callback=CustomCallback,
                cv_count=cv_count,
                common_config_modification_func=ds_ctx_common_config_settings,
                common_callback_modification_func=common_callback_func,
                custom_config_modification_func=ctx_cnn_custom_config)


if __name__ == "__main__":
    run_testing_cnn()
