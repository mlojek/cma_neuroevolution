# CMA-ES Neuroevolution
This project probes the possibility and feasibility of training neural networks with CMA-ES algorithm.

Artificial neural networks play a crucial role in modern machine learning. These deep learning models are mostly trained using gradient methods, e.g. SGD and ADAM. These methods however, are not the only ways of finding optimal parameter values. Neural networks can be trained using genetic algorithms - such process is called neuroevolution. This project explores the viablity of using CMA-ES algorithm to optimize simple neural networks. CMA-ES is the state-of-the-art black-box optimization which optimizes a function by randomly sampling the search space with a multivariate normal distribution. 

## Installation
For this project, python 3.13 is recommended. To install dependencies run:
```bash
make install
# OR
pip install -r requirements.txt
```
This project uses wandb.ai for experiment tracking. During model training, the train and val average loss and accuracy are logged to the service. To enable wandb.ai in this project, run:
```bash
export WANDB_API_KEY=<your_api_key>
wandb login
```

## Usage instructions
Before running the project a configs file needs to be created. Example configs can be found in `configs/` directory. To run an experiment run:
```bash
python experiment.py path/to/config
```

## Experiments
To compare the gradient optimizers with CMA-ES, the following experiment setup will be used:
- Datasets: Iris, MNIST (downscaled to 14x14).
- Gradient optimizers: SGD, Adam.
- CMA-ES strategies: one optimizer for all model parameters, one optimizer per model layer.
- Performance metrics: crossentropy loss, accuracy on test set.
- Efficiency metrics: training time in seconds, model evaluations, gradient calculations.


## Results

TODO description
- all params except lr and popsize stay the same
- first for every method the optimal value will be selected (based on test loss)
- early stopping' there
- 25 runs per experiment

### Iris dataset

TODO link config

| Training mode    | Model evals | Gradient evals | Train time | Test loss | Test accuracy |
|------------------|-------------|----------------|------------|-----------|---------------|
| SGD              | 1535        | 69.3           | 0.05s      | 0.649     | 0.91          |
| ADAM             | 1583        | 71.6           | 0.06s      | 0.615     | 0.935         |
| CMA-ES           | 13321       | 0              | 8.54s      | 0.659     | 0.892         |
| LAYERWISE CMA-ES | 36020       | 0              | 1.15s      | 0.706     | 0.852         |    

adam best, sgd comparable, cmaes worse and less efficient

### MNIST dataset

| Training mode    | Model evals | Gradient evals | Train time | Test loss | Test accuracy |
|------------------|-------------|----------------|------------|-----------|---------------|
| SGD              |        |            |       | 0.    | 0.          |
| ADAM             |         |             |        | 0.    | 0.          |
| CMA-ES           |        |               |       | 0.    | 0.          |
| LAYERWISE CMA-ES |        |               |       | 0.    | 0.          |    


## References