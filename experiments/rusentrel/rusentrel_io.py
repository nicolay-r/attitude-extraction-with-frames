import collections
import os
import io_utils
import utils as e_utils

from core.common.utils import create_dir_if_not_exists
from core.networks.network_io import NetworkIO
from core.networks.data_type import DataType
from core.source.rusentrel.helpers.parsed_news import RuSentRelParsedNewsHelper
from core.source.rusentrel.news import RuSentRelNews
from core.source.rusentrel.io_utils import RuSentRelIOUtils
from core.source.rusentrel.entities.collection import RuSentRelDocumentEntityCollection
from core.source.rusentrel.opinions.collection import RuSentRelOpinionCollection
from core.source.rusentrel.synonyms import RuSentRelSynonymsCollection
from core.evaluation.utils import OpinionCollectionsToCompareUtils
from core.processing.lemmatization.base import Stemmer

from experiments.rusentrel.rusvectores_io import get_rusvectores_news_embedding_filepath
from experiments.rusentrel.neutrals_io import RuSentRelNeutralIOUtils
from experiments.rusentrel.utils import get_cv_pair_by_index


class RuSentRelNetworkIO(NetworkIO):
    """
    Represents Input interface for NeuralNetwork ctx
    Now exploited (treated) as input interface only
    """

    def __init__(self, model_name, cv_count=1):
        assert(isinstance(cv_count, int))
        self.__synonyms = None
        self.__cv_count = cv_count
        self.__current_cv_index = 0
        self.__model_name = model_name

    # region properties

    @property
    def SynonymsCollection(self):
        return self.__synonyms

    @property
    def CVCount(self):
        return self.__cv_count

    @property
    def CVCurrentIndex(self):
        return self.__current_cv_index

    # endregion

    def inc_cv_index(self):
        self.__current_cv_index += 1

    # region 'get' public methods

    def get_word_embedding_filepath(self):
        return get_rusvectores_news_embedding_filepath()

    def get_model_filepath(self):
        return os.path.join(self.__get_model_states_dir(),
                            u'{}'.format(self.__model_name))

    def get_model_root(self):
        return self.__get_model_root()

    @staticmethod
    def get_capitals_filepath():
        return os.path.join(io_utils.get_data_root(), u'capitals.lss')

    @staticmethod
    def get_states_filepath():
        return os.path.join(io_utils.get_data_root(), u'states.lss')

    # endregion

    # region 'write' methods

    def write_log(self, log_names, log_values):
        assert(isinstance(log_names, list))
        assert(isinstance(log_values, list))
        assert(len(log_names) == len(log_values))

        log_path = os.path.join(self.get_model_root(), u"log.txt")

        with open(log_path, 'w') as f:
            for index, log_value in enumerate(log_values):
                f.write("{}: {}\n".format(log_names[index], log_value))

    # endregion

    # region 'read' public methods

    def read_parsed_news(self, doc_id, keep_tokens, stemmer):
        assert(isinstance(doc_id, int))
        assert(isinstance(keep_tokens, bool))
        assert(isinstance(stemmer, Stemmer))

        entities = RuSentRelDocumentEntityCollection.read_collection(doc_id=doc_id,
                                                                     stemmer=stemmer,
                                                                     synonyms=self.__synonyms)

        news = RuSentRelNews.read_document(doc_id, entities)

        parsed_news = RuSentRelParsedNewsHelper.create_parsed_news(rusentrel_news_id=doc_id,
                                                                   rusentrel_news=news,
                                                                   keep_tokens=keep_tokens,
                                                                   stemmer=stemmer)

        return news, parsed_news

    def read_synonyms_collection(self, stemmer):
        assert(isinstance(stemmer, Stemmer))
        self.__synonyms = RuSentRelSynonymsCollection.read_collection(stemmer=stemmer,
                                                                      is_read_only=True)

    def read_neutral_opinion_collection(self, doc_id, data_type):
        assert(isinstance(data_type, unicode))

        filepath = RuSentRelNeutralIOUtils.get_rusentrel_neutral_opin_filepath(
            doc_id=doc_id,
            is_train=True if data_type == DataType.Train else False)

        return RuSentRelOpinionCollection.read_from_file(filepath=filepath,
                                                         synonyms=self.__synonyms)

    def read_etalon_opinion_collection(self, doc_id):
        assert(isinstance(doc_id, int))

        return RuSentRelOpinionCollection.read_collection(doc_id=doc_id,
                                                          synonyms=self.__synonyms)

    @staticmethod
    def read_list_from_lss(filepath):
        """
        Reading lines in lowercase mode
        """
        lines = []
        with open(filepath) as f:
            for line in f.readlines():
                row = line.decode('utf-8')
                row = row.lower().strip()
                lines.append(row)

        return lines

    # endregion

    # region 'iters' public methods

    def iter_data_indices(self, data_type):
        if data_type == DataType.Train:
            return self.iter_train_data_indices()
        if data_type == DataType.Test:
            return self.iter_test_data_indices()

    def iter_test_data_indices(self):
        if self.__cv_count == 1:
            for doc_id in RuSentRelIOUtils.iter_test_indices():
                yield doc_id
        else:
            _, test = get_cv_pair_by_index(cv_count=self.__cv_count,
                                           cv_index=self.__current_cv_index)
            for doc_id in test:
                yield doc_id

    def iter_train_data_indices(self):
        if self.__cv_count == 1:
            for doc_id in RuSentRelIOUtils.iter_train_indices():
                yield doc_id
        else:
            train, _ = get_cv_pair_by_index(cv_count=self.__cv_count,
                                            cv_index=self.__current_cv_index)
            for doc_id in train:
                yield doc_id

    def iter_opinion_collections_to_compare(self, data_type, doc_ids, epoch_index):
        assert(isinstance(data_type, unicode))
        assert(isinstance(doc_ids, collections.Iterable))
        assert(isinstance(epoch_index, int))

        it = OpinionCollectionsToCompareUtils.iter_comparable_collections(
            doc_ids=doc_ids,
            read_etalon_collection_func=lambda doc_id: self.read_etalon_opinion_collection(doc_id=doc_id),
            read_result_collection_func=lambda doc_id: RuSentRelOpinionCollection.read_from_file(
                filepath=self.create_result_opinion_collection_filepath(data_type=data_type,
                                                                        doc_id=doc_id,
                                                                        epoch_index=epoch_index),
                synonyms=self.__synonyms))

        for opinions_cmp in it:
            yield opinions_cmp

    # endregion

    # region 'create' public methods

    def create_opinion_collection(self):
        return RuSentRelOpinionCollection([], synonyms=self.__synonyms)

    def create_model_state_filepath(self):
        return os.path.join(self.__get_model_states_dir(),
                            u'{}.state'.format(self.__model_name))

    def create_result_opinion_collection_filepath(self, data_type, doc_id, epoch_index):
        assert(isinstance(epoch_index, int))

        model_eval_root = self.__get_eval_root_filepath(data_type=data_type,
                                                        epoch_index=epoch_index)

        filepath = os.path.join(model_eval_root, u"{}.opin.txt".format(doc_id))
        create_dir_if_not_exists(filepath)
        return filepath

    # endregion

    # region private methods

    def __get_model_root(self):
        return e_utils.get_path_of_subfolder_in_experiments_dir(self.__model_name)

    def __get_model_states_dir(self):

        result_dir = os.path.join(
            self.__get_model_root(),
            os.path.join(u'model_states/'))

        create_dir_if_not_exists(result_dir)
        return result_dir

    def __get_eval_root_filepath(self, data_type, epoch_index):
        assert(isinstance(epoch_index, int))

        result_dir = os.path.join(
            self.__get_model_root(),
            os.path.join(u"eval/{}/{}/{}".format(data_type,
                                                 self.__current_cv_index,
                                                 str(epoch_index))))

        create_dir_if_not_exists(result_dir)
        return result_dir

    # endregion
