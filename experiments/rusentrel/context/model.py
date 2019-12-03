from experiments.rusentrel.context.helpers.initialization import ContextModelInitHelper
from experiments.rusentrel.rusentrel_io import RuSentRelNetworkIO

from core.networks.eval.opinion_based import OpinionBasedEvaluationHelper
from core.source.rusentrel.opinions.collection import RuSentRelOpinionCollection
from core.networks.context.configurations.base import DefaultNetworkConfig
from core.networks.context.training.batch import MiniBatch
from core.networks.callback import Callback
from core.networks.data_type import DataType
from core.networks.tf_model import TensorflowModel
from core.networks.network import NeuralNetwork


class ContextLevelTensorflowModel(TensorflowModel):

    def __init__(self, io, network, config, evaluator_class, callback):
        assert(isinstance(io, RuSentRelNetworkIO))
        assert(isinstance(config, DefaultNetworkConfig))
        assert(isinstance(network, NeuralNetwork))
        assert(isinstance(callback, Callback) or callback is None)
        assert(callable(evaluator_class))

        super(ContextLevelTensorflowModel, self).__init__(
            io=io, network=network, callback=callback)

        self.__config = config
        self.__evaluator_class = evaluator_class
        self.__init_helper = None
        self.__eval_helper = None
        self.prepare_sources()

    @property
    def Config(self):
        return self.__config

    def prepare_sources(self):
        self.__init_helper = self.create_model_init_helper()
        # TODO. In core
        self.__eval_helper = OpinionBasedEvaluationHelper(
            self.__evaluator_class(synonyms=self.IO.SynonymsCollection))
        self.__print_statistic()

    def get_bags_collection(self, data_type):
        return self.__init_helper.BagsCollections[data_type]

    def get_bags_collection_helper(self, data_type):
        return self.__init_helper.BagsCollectionHelpers[data_type]

    def get_text_opinions_collection(self, data_type):
        return self.__init_helper.TextOpinionCollections[data_type]

    def get_text_opinions_collection_helper(self, data_type):
        return self.__init_helper.TextOpinionCollectionHelpers[data_type]

    def get_gpu_memory_fraction(self):
        return self.__config.GPUMemoryFraction

    def get_labels_helper(self):
        return self.__init_helper.LabelsHelper

    def get_eval_helper(self):
        return self.__eval_helper

    def before_evaluation(self, dest_data_type):
        helper = self.get_text_opinions_collection_helper(dest_data_type)

        helper.debug_labels_statistic()

        collections_iter = helper.iter_opinion_collections(
            create_collection_func=lambda: self.IO.create_opinion_collection(),
            label_calculation_mode=self.Config.TextOpinionLabelCalculationMode)

        for collection, news_id in collections_iter:
            assert(isinstance(collection, RuSentRelOpinionCollection))
            filepath = self.IO.create_result_opinion_collection_filepath(data_type=dest_data_type,
                                                                         doc_id=news_id,
                                                                         epoch_index=self.CurrentEpochIndex)
            collection.save_to_file(filepath)

    def create_batch_by_bags_group(self, bags_group):
        return MiniBatch(bags_group)

    def create_model_init_helper(self):
        return ContextModelInitHelper(io=self.IO, config=self.Config)

    def __print_statistic(self):
        keys, values = self.Config.get_parameters()
        self.IO.write_log(log_names=keys, log_values=values)
        self.get_text_opinions_collection_helper(DataType.Train).debug_labels_statistic()
        self.get_text_opinions_collection_helper(DataType.Train).debug_unique_relations_statistic()
        self.get_text_opinions_collection_helper(DataType.Test).debug_labels_statistic()
        self.get_text_opinions_collection_helper(DataType.Test).debug_unique_relations_statistic()
        self.get_bags_collection_helper(DataType.Train).print_log_statistics()
        self.get_bags_collection_helper(DataType.Test).print_log_statistics()

