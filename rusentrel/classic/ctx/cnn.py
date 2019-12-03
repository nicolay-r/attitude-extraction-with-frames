#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import tensorflow as tf

sys.path.append('../../../')

from core.networks.context.configurations.cnn import CNNConfig
from core.networks.context.architectures.cnn import VanillaCNN
from core.evaluation.evaluators.two_class import TwoClassEvaluator

from rusentrel.ctx_names import ModelNames
from rusentrel.utils import run_testing
from rusentrel.callback import CustomCallback

from experiments.rusentrel.rusentrel_io import RuSentRelNetworkIO
from experiments.rusentrel.context.model import ContextLevelTensorflowModel

from rusentrel.classic.common import \
    classic_ctx_common_config_settings, \
    classic_common_callback_modification_func


def ctx_cnn_custom_config(config):
    assert(isinstance(config, CNNConfig))
    config.modify_weight_initializer(tf.contrib.layers.xavier_initializer())


def run_testing_cnn(cv_count=1,
                    name_prefix=u'',
                    custom_config_func=ctx_cnn_custom_config,
                    custom_callback_func=classic_common_callback_modification_func):

    run_testing(model_name=name_prefix + ModelNames.CNN,
                create_network=VanillaCNN,
                create_config=CNNConfig,
                create_io=RuSentRelNetworkIO,
                create_model=ContextLevelTensorflowModel,
                cv_count=cv_count,
                evaluator_class=TwoClassEvaluator,
                create_callback=CustomCallback,
                common_callback_modification_func=custom_callback_func,
                custom_config_modification_func=custom_config_func,
                common_config_modification_func=classic_ctx_common_config_settings)


if __name__ == "__main__":

    run_testing_cnn()

