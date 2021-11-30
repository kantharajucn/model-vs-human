# Model training on model-vs-human Datasets

This repository is to train Pytorch models on all the 17 datasets from [model-vs-human](https://github.com/bethgelab/model-vs-human).  
Basic idea is that we train only the last layer(classifier) by freezing all other layers on 17 datasets that were using in `m̀odel-vs-human`.

## How to run

1. Set the environment variable `M̀ODEL_VS_HUMAN_DIR`
```shell
export MODEL_VS_HUMAN_DIR=/path/to/model-vs-human-training
```

2. Download all the datasets by running the script `dataset_downloader.py`
```shell

    python3 dataset_download.py

```

All the dataset will be downloaded to the directory `$(MODEL_VS_HUMAN)/datasets` path.`

3. Split the datasets

This step will split all 17 datasets into a single train and test sets in the ratio of 0.5:0.5. 
Once the splitting is completed, `train.csv`and `test.csv` will be saved under `$(MODEL_VS_HUMAN_DIR/input`

```shell
python3 split_dataset.py
```
4. Model training

Train any of the model that is available in Pytorch model_zoo. Below is the sample command to train model `squeezenet1_1`. 

```shell
python3 train.py --batch-size=16 --arch=squeezenet1_1
```

## How to add model initializers

It is necessary to freeze all the hidden layers and change the classifier to match to our dataset with only 16 classes.
There are already few initializers are added in it but you should add one if the model you are planning to train doesn't exists yet.

