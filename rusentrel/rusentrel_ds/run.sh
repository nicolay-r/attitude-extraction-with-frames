#!/usr/bin/env bash
source ../../../erc-v6/my_env/bin/activate

pushd .

cd ctx
python -u ian_frames.py > __ian_frames.txt
python -u ian_ends.py > __ian_ends.txt
python -u att_cnn.py > __att-cnn-log.txt
python -u att_pcnn.py > __att-pcnn-log.txt
python -u att_frames_cnn.py > __att-frames-cnn-log.txt
python -u att_frames_pcnn.py > __att-frames-pcnn-log.txt
python -u att_bilstm.py > __att-bilstm-log.txt
python -u self_att_bilstm.py > __self-att-bilstm-log.txt
python -u cnn.py > __cnn-log.txt
python -u pcnn.py > __pcnn-log.txt
python -u lstm.py > __lstm-log.txt
python -u bilstm.py > __bilstm-log.txt
python -u rcnn.py > __rcnn-log.txt
popd

cd mi
python -u ian_frames.py > __ian_frames.txt
python -u ian_ends.py > __ian_ends.txt
python -u att_cnn.py > __att-cnn-log.txt
python -u att_pcnn.py > __att-pcnn-log.txt
python -u att_frames_cnn.py > __att-frames-cnn-log.txt
python -u att_frames_pcnn.py > __att-frames-pcnn-log.txt
python -u att_bilstm.py > __att-bilstm-log.txt
python -u self_att_bilstm.py > __self-att-bilstm-log.txt
python -u cnn.py > __cnn-log.txt
python -u pcnn.py > __pcnn-log.txt
python -u lstm.py > __lstm-log.txt
python -u bilstm.py > __bilstm-log.txt
python -u rcnn.py > __rcnn-log.txt
popd
