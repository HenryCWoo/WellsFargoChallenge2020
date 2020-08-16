# WellsFargoChallenge2020
This is my submission to this challenge: [MindSumo Wells Fargo Campus Analytics Challenge](https://www.mindsumo.com/contests/campus-analytics-challenge-2020).  

This project contains models that can be trained via the command line interface.
There are a total of four modifiable neural network architectures and a bare minimum usage of XGBoost.  

All experiments use the same seed which makes them reproducible.

## Directories
`/notebooks`  
Jupyter notebooks used for exploratory analyses of the data.

`/src`  
Implementation for training, evaluating, and testing models.

`/src/models`  
Implementation of the models written in PyTorch.

`/src/scripts`  
Scripts for aggregating experiment results and grid searching.

`/reports`  
Directory that stores the aggregated experiment results and other results.

`/experiments`  
Auto-generated directory that stores model parameter descriptions, results, and frozen models. All experiments are identified by a unique number and will contain a file called `info.yml` that records the results and model hyperparameters used from the command line interface.
Frozen models are named `model.sav`.

## Installation
Use anaconda to create an environment using:

```bash
conda create --name <env> --file requirements.txt
```

## Usage
Before all else, first create the directories `/data/raw` in the root directory.  
Then place both the following datasets/files:
- 0173eeb640e7-Challenge+Data+Set+-+Campus+Analytics+2020.xlsx
- d59675225279-Evaluation+Data+Set+-+Campus+Analytics+2020.xlsx  
inside `/data/raw`.

The training sessions, hyperparameters, and modifications to the architectures can all be done from the command line interface.  
All commands below assume the user's current working directory is `/src`.

Use the following to view all flags that modify the training session.
```bash
python train.py -h
```

For example, the following command produces the best results I personally came across through the grid search method.
```bash
python train.py --model=conv1d_embed --lr=1e-4 --epochs=128 --conv_blocks=1 --filters=128 --hidden_layer=3 --hidden_units=64 --kernel_size=2
```
After the experiments finishes, search for the experiment in the `/experiments` folder and see `info.yml` to see results and the hyperparameters used from above.

To generate results on the evaluation set provided from the contest, use the following command:
```bash
python train.py --test --exp_no=<EXPERIMENT NUMBER FOUND IN /experiments>
```
This will create a file called `output.csv` inside the experiment number directory. The file will contain the dataset_id and prediction_score columns as specified in the contest description.

## Contributing
As this is a single-time contest entry, contributions are not accepted. 

All work was done solely by me, Henry C Woo.
