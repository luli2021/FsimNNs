This repository contains the implementation and resources for our anonymous submission to ICCAD 2025, titled:

> FsimNNs: An Open-Source Graph Neural Network Platform for SEU Simulation-based Fault Injection



## Requirements
* Python >= 3.9 
* PyTorch >= 2.6 
* PyTorch Geometric
* pytorch_lightning
* pandas
* scikit-learn


## Usage
1. upzip the data
2. cd stgnns; change the setting in *.yaml
3. use `python train.py --config train_config.yaml` to run the training.
4. use `python predict.py --config predict_config.yaml` to run the prediction.
