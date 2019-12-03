from core.common.opinions.collection import OpinionCollection
from core.processing.lemmatization.base import Stemmer
from core.source.ruattitudes.helpers.news_helper import RuAttitudesNewsHelper
from core.source.ruattitudes.helpers.parsed_news import RuAttitudesParsedNewsHelper
from core.source.ruattitudes.news import RuAttitudesNews
from core.source.ruattitudes.reader import RuAttitudesFormatReader
from core.source.rusentrel.io_utils import RuSentRelIOUtils
from ..rusentrel.rusentrel_io import RuSentRelNetworkIO


class RuSentRelWithRuAttitudesIO(RuSentRelNetworkIO):

    def __init__(self, model_name, cv_count):
        super(RuSentRelWithRuAttitudesIO, self).__init__(model_name=model_name,
                                                         cv_count=cv_count)
        self.__rusentrel_ids = list(RuSentRelIOUtils.iter_collection_indices())
        self.__ru_atttudes = None

    # region 'read' public methods

    def read_synonyms_collection(self, stemmer):
        super(RuSentRelWithRuAttitudesIO, self).read_synonyms_collection(stemmer)
        print "Loading RuAttitudes collection in memory, please wait ..."
        self.__ru_atttudes = self.__read_ruattitudes_in_memory(stemmer)

    def read_parsed_news(self, doc_id, keep_tokens, stemmer):
        if doc_id in self.__rusentrel_ids:
            return super(RuSentRelWithRuAttitudesIO, self).read_parsed_news(doc_id=doc_id,
                                                                            keep_tokens=keep_tokens,
                                                                            stemmer=stemmer)

        news = self.__ru_atttudes[doc_id]
        parsed_news = RuAttitudesParsedNewsHelper.create_parsed_news(doc_id=doc_id,
                                                                     news=news)

        return news, parsed_news

    def read_etalon_opinion_collection(self, doc_id):
        assert(isinstance(doc_id, int))

        if doc_id in self.__rusentrel_ids:
            return super(RuSentRelWithRuAttitudesIO, self).read_etalon_opinion_collection(doc_id)

        news = self.__ru_atttudes[doc_id]
        opinions = [opinion for opinion, _ in RuAttitudesNewsHelper.iter_opinions_with_related_sentences(news)]

        return OpinionCollection(opinions=opinions,
                                 synonyms=self.SynonymsCollection)

    def read_neutral_opinion_collection(self, doc_id, data_type):
        assert(isinstance(doc_id, int))
        assert(isinstance(data_type, unicode))

        if doc_id in self.__rusentrel_ids:
            return super(RuSentRelWithRuAttitudesIO, self).read_neutral_opinion_collection(doc_id, data_type)

        # TODO. Complete.
        pass

    # endregion

    # region private methods

    @staticmethod
    def __read_ruattitudes_in_memory(stemmer):
        assert(isinstance(stemmer, Stemmer))

        d = {}
        for news in RuAttitudesFormatReader.iter_news(stemmer=stemmer):
            assert(isinstance(news, RuAttitudesNews))
            d[news.NewsIndex] = news

        return d

    def iter_train_data_indices(self):
        for doc_id in super(RuSentRelWithRuAttitudesIO, self).iter_train_data_indices():
            yield doc_id
        for doc_id in self.__ru_atttudes.iterkeys():
            yield doc_id

    # endregion
