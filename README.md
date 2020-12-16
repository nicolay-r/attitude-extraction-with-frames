# attitude-extraction-with-frames
![](https://img.shields.io/badge/Python-2.7-brightgreen.svg)

> Description to be updated.

# Contents
* [Dependencies](#dependencies)
* [Installation](#installation)
* [Experiments Reproduction](#how-to-run-experiments)

## Dependencies

List of the toolset dependencies are as follows:

* Python 2.7

* [AREkit](https://github.com/nicolay-r/AREkit) -- a toolset for sentiment attitudes extraction task 
(checkout [Installation](#installation) section for details); 

## Installation 

Includes three steps:

* Clone this repository:
```
git clone https://github.com/nicolay-r/attitude-extraction-with-frames
```

* install [AREkit](https://github.com/nicolay-r/AREkit) 
as a dependency in `core` directory:
```
cd attitude-extraction-with-frames
git clone https://github.com/nicolay-r/AREkit/tree/0.19.3-lrec-rejected-rc core
```

* install data dependencies:
```
cd data
./install.sh
```

## How to Reproduce Experiments

* Select the experiment type:
    * `classic` -- is an application of 
        [RuSentRel](https://github.com/nicolay-r/RuSentRel) 
        only for training.
    * `rusentrel_ds` -- is an application of [RuSentRel](https://github.com/nicolay-r/RuSentRel) and [RuAttitudes](https://github.com/nicolay-r/RuAttitudes) 
        datasets for training.
```
# experiment type #1
cd rusentrel/classic/ctx
# experiment type #2
cd rusentrel/rusentrel_ds/ctx
```

* Run one of the following experiment:
```
python bilstm.py
python cnn.py
python lstm.py
python pcnn.py
```
