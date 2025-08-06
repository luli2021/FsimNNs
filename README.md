This repository contains the implementation and resources for our paper, titled:

> FsimNNs: An Open-Source Graph Neural Network Platform for SEU Simulation-based Fault Injection

## Requirements
* Python >= 3.9 (3.12 recommended)
* PyTorch >= 2.5 
* PyTorch Geometric
* pytorch_lightning
* pandas
* scikit-learn


## Usage
1. upzip the data.zip to get the data of SEU fault simulation.
2. cd stgnns; change the configurations in *.yaml.
   * modify the directories of the dataset;
   * make sure the number of nodes (num_nodes) in the dataset is correct, you can find the information in dataset.txt in the data;
   * adjust the hyperparameters as you want. 
3. use `python train.py --config train_config.yaml` to run the training.
4. use `python predict.py --config predict_config.yaml` to run the prediction.

## Folder Structure
  ```
  stgnns/
  │
  ├── train.py - main script to start training
  ├── predict.py - evaluation on the test dataset
  │
  ├── train_config.yaml - configuration for training
  ├── prediction_config.yaml - configuration for prediction
  │
  ├── datasets/ - to generate datasets of PyG
  │   ├── customized_dataset.py
  │   ├── data_preprocessing.py
  │   └── data_splitting.py
  │
  ├── model/ - models 
  │   ├── models.py
  │   └── gnnmodels/
  |       ├── stgcn.py -STGCN
  |       ├── aspp_stgat.py -ASTGCN
  │       └── aspp_stgat.py -ASTGAT
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ```

