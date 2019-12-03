#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys

sys.path.append('../../../')

from core.networks.context.architectures.pcnn import PiecewiseCNN
from core.networks.context.configurations.cnn import CNNConfig
from core.evaluation.evaluators.two_class import TwoClassEvaluator

from rusentrel.callback import CustomCallback
from rusentrel.utils import run_testing
from rusentrel.ctx_names import ModelNames
from rusentrel.rusentrel_ds.common import DS_NAME_PREFIX, \
    ds_ctx_common_config_settings, \
    ds_common_callback_modification_func
from rusentrel.classic.ctx.pcnn import ctx_pcnn_custom_config

from experiments.rusentrel_ds.rusentrel_ds_io import RuSentRelWithRuAttitudesIO
from experiments.rusentrel.context.model import ContextLevelTensorflowModel


def run_testing_pcnn(
        name_prefix=DS_NAME_PREFIX,
        cv_count=1,
        common_callback_func=ds_common_callback_modification_func):

    run_testing(model_name=name_prefix + ModelNames.PCNN,
                create_network=PiecewiseCNN,
                create_config=CNNConfig,
                cv_count=cv_count,
                create_io=RuSentRelWithRuAttitudesIO,
                create_model=ContextLevelTensorflowModel,
                evaluator_class=TwoClassEvaluator,
                create_callback=CustomCallback,
                common_config_modification_func=ds_ctx_common_config_settings,
                common_callback_modification_func=common_callback_func,
                custom_config_modification_func=ctx_pcnn_custom_config)


if __name__ == "__main__":

    run_testing_pcnn()


