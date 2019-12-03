#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys


sys.path.append('../../../')

from core.networks.context.configurations.bi_lstm import BiLSTMConfig
from core.networks.context.architectures.bilstm import BiLSTM
from core.networks.tf_helpers.sequence import CellTypes
from core.evaluation.evaluators.two_class import TwoClassEvaluator

from rusentrel.ctx_names import ModelNames
from rusentrel.utils import run_testing
from rusentrel.callback import CustomCallback

from experiments.rusentrel.rusentrel_io import RuSentRelNetworkIO
from experiments.rusentrel.context.model import ContextLevelTensorflowModel

from rusentrel.classic.common import \
    classic_ctx_common_config_settings, \
    classic_common_callback_modification_func


def ctx_bilstm_custom_config(config):
    assert(isinstance(config, BiLSTMConfig))
    config.modify_hidden_size(128)
    config.modify_bags_per_minibatch(2)
    config.modify_cell_type(CellTypes.BasicLSTM)
    config.modify_dropout_rnn_keep_prob(0.8)
    config.modify_bag_size(2)
    config.modify_bags_per_minibatch(4)
    config.modify_terms_per_context(25)


def run_testing_bilstm(name_prefix=u'',
                       cv_count=1,
                       custom_config_func=ctx_bilstm_custom_config,
                       custom_callback_func=classic_common_callback_modification_func):

    run_testing(model_name=name_prefix+ModelNames.BiLSTM,
                create_network=BiLSTM,
                create_config=BiLSTMConfig,
                create_model=ContextLevelTensorflowModel,
                create_io=RuSentRelNetworkIO,
                cv_count=cv_count,
                create_callback=CustomCallback,
                evaluator_class=TwoClassEvaluator,
                common_callback_modification_func=custom_callback_func,
                custom_config_modification_func=custom_config_func,
                common_config_modification_func=classic_ctx_common_config_settings)


if __name__ == "__main__":

    run_testing_bilstm()
