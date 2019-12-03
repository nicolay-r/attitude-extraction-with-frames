#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys


sys.path.append('../../../')

from core.networks.context.configurations.bi_lstm import BiLSTMConfig
from core.networks.context.architectures.bilstm import BiLSTM
from core.evaluation.evaluators.two_class import TwoClassEvaluator

from rusentrel.ctx_names import ModelNames
from rusentrel.utils import run_testing
from rusentrel.callback import CustomCallback
from rusentrel.rusentrel_ds.common import DS_NAME_PREFIX, \
    ds_ctx_common_config_settings, \
    ds_common_callback_modification_func
from rusentrel.classic.ctx.bilstm import ctx_bilstm_custom_config

from experiments.rusentrel_ds.rusentrel_ds_io import RuSentRelWithRuAttitudesIO
from experiments.rusentrel.context.model import ContextLevelTensorflowModel


def run_testing_bilstm(
        name_prefix=DS_NAME_PREFIX,
        cv_count=1,
        common_callback_func=ds_common_callback_modification_func):

    run_testing(model_name=name_prefix + ModelNames.BiLSTM,
                create_network=BiLSTM,
                create_config=BiLSTMConfig,
                create_model=ContextLevelTensorflowModel,
                create_io=RuSentRelWithRuAttitudesIO,
                cv_count=cv_count,
                create_callback=CustomCallback,
                evaluator_class=TwoClassEvaluator,
                common_config_modification_func=ds_ctx_common_config_settings,
                common_callback_modification_func=common_callback_func,
                custom_config_modification_func=ctx_bilstm_custom_config)


if __name__ == "__main__":
    run_testing_bilstm()
