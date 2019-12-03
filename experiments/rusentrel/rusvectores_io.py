from os import path
from io_utils import get_data_root


def get_rusvectores_news_embedding_filepath():
    return path.join(get_data_root(), u"w2v/news_rusvectores2.bin.gz")