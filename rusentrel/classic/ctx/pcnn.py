#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import tensorflow as tf


sys.path.append('../../../')

from core.networks.context.architectures.pcnn import PiecewiseCNN
from core.networks.context.configurations.cnn import CNNConfig
from core.evaluation.evaluators.two_class import TwoClassEvaluator

from rusentrel.callback import CustomCallback
from rusentrel.utils import run_testing
from rusentrel.ctx_names import ModelNames

from experiments.rusentrel.rusentrel_io import RuSentRelNetworkIO
from experiments.rusentrel.context.model import ContextLevelTensorflowModel

from rusentrel.classic.common import \
    classic_ctx_common_config_settings, \
    classic_common_callback_modification_func


def ctx_pcnn_custom_config(config):
    assert(isinstance(config, CNNConfig))
    config.modify_bags_per_minibatch(2)
    config.modify_weight_initializer(tf.contrib.layers.xavier_initializer())


def run_testing_pcnn(cv_count=1,
                     name_prefix=u'',
                     custom_config_func=ctx_pcnn_custom_config,
                     custom_callback_func=classic_common_callback_modification_func):

    run_testing(model_name=name_prefix + ModelNames.PCNN,
                create_network=PiecewiseCNN,
                create_config=CNNConfig,
                cv_count=cv_count,
                create_io=RuSentRelNetworkIO,
                create_model=ContextLevelTensorflowModel,
                evaluator_class=TwoClassEvaluator,
                create_callback=CustomCallback,
                common_callback_modification_func=custom_callback_func,
                custom_config_modification_func=custom_config_func,
                common_config_modification_func=classic_ctx_common_config_settings)


if __name__ == "__main__":

    run_testing_pcnn()
