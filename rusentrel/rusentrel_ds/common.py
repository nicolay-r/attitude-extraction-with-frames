from core.networks.context.configurations.base import DefaultNetworkConfig
from core.networks.multi.configuration.base import BaseMultiInstanceConfig
from rusentrel.callback import CustomCallback
from rusentrel.classic.common import classic_ctx_common_config_settings
from rusentrel.default import MI_CONTEXTS_PER_OPINION

DS_NAME_PREFIX = u'ds_'


def ds_ctx_common_config_settings(config):
    """
    This function describes a base config setup for all models.
    """
    assert(isinstance(config, DefaultNetworkConfig))
    config.modify_test_on_epochs(range(0, 50, 5))

    # Apply classic settings
    classic_ctx_common_config_settings(config)

    # Increasing memory limit consumption
    config.modify_gpu_memory_fraction(0.65)


def ds_common_callback_modification_func(callback):
    """
    This function describes configuration setup for all model callbacks.
    """
    assert(isinstance(callback, CustomCallback))
    callback.set_key_perform_train_evaluation(False)
    callback.set_cancellation_acc_bound(0.99)
    callback.set_key_stop_training_by_cost(False)

