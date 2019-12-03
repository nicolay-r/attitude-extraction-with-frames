# Download Embedding
$w2v_dir=w2v
mkdir -p $w2v_dir
curl http://rusvectores.org/static/models/rusvectores2/news_mystem_skipgram_1000_20_2015.bin.gz -o $w2v_dir"news_rusvectores2.bin.gz"
