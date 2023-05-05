# Distributed Nystroem Approximation
## Intro
This repository provides the code used to run the experiments of the JMLR paper "Optimal Convergence Rates for Distributed Nystroem Approximation".
## Environments
- Python 3.9.15
- Pytorch 1.13.0
- [NNI 2.10](https://github.com/microsoft/nni)
- CUDA 11.7
- cuDnn 8.2.1
- GPU: Nvidia RTX 2080Ti 11GB
## Core functions
- functions/algorithms.py implements all compared methods.
- functions/experiments.py construct repeatable experiments in the paper.
- functions/optimal_parameters.py records optimal parameters for the proposed algorithm.
- functions/utils.py defines data loaders and evaluation measures.
- tune_hyperparameter.py uses NNI framework to tune the optimal hyperparameters.
## Experiments
1. Download datasets from [UCI datasets](http://archive.ics.uci.edu/ml/index.php).
2. Run the script to tune parameters via NNI and record them in optimal_parameters.py.
```
python tune_hyperparameter.py
```
3. Run the script to obtain results in Experiment section, which will be saved in `results` folder.
```
python runscripts.py
```
4. Use `plot.ipynb` to draw figures from the experiment results.