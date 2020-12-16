from core.networks.context.configurations.base import DefaultNetworkConfig


def classic_ctx_common_config_settings(config):
    """
    Context version
    """
    assert(isinstance(config, DefaultNetworkConfig))

    config.modify_test_on_epochs(range(0, 151, 5))
    config.modify_classes_count(3)
    config.modify_learning_rate(0.1)
    config.modify_use_class_weights(True)
    config.modify_dropout_keep_prob(0.5)
    config.modify_bag_size(1)
    config.modify_bags_per_minibatch(8)
    config.modify_gpu_memory_fraction(0.4)
    config.modify_embedding_dropout_keep_prob(1.0)
    config.modify_terms_per_context(50)


def classic_common_callback_modification_func(callback):
    """
    This function describes configuration setup for all model callbacks.
    """
    callback.set_key_stop_training_by_cost(False)
