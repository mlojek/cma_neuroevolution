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
- Datasets: Iris, MNIST.
- Gradient optimizers: SGD, Adam.
- CMA-ES strategies: one optimizer for all model parameters, one optimizer per model layer.
- Performance metrics: crossentropy loss, accuracy on test set.
- Efficiency metrics: training time in seconds, model evaluations, gradient calculations.


## Results
TODO description

### Iris dataset

| Training mode    | Model evals | Gradient evals | Train time | Test loss | Test accuracy |
|------------------|-------------|----------------|------------|-----------|---------------|
| SGD              | 24000       | 1200           | 2.33s      | 0.6220    | 0.93          |
| ADAM             | 4800        | 240            | 1.8s       | 0.6222    | 0.93          |
| CMA-ES           | 14100       | 0              | 4.17s      | 0.7039    | 0.83          |
| LAYERWISE CMA-ES | 15360       | 0              | 1.35s      | 0.6177    | 0.93          |    


### MNIST dataset

| Training mode    | Model evals | Gradient evals | Train time | Test loss | Test accuracy |
|------------------|-------------|----------------|------------|-----------|---------------|
| SGD              | 24000       | 1200           | 2.33s      | 0.6220    | 0.93          |
| ADAM             | 4800        | 240            | 1.8s       | 0.6222    | 0.93          |
| CMA-ES           | 14100       | 0              | 4.17s      | 0.7039    | 0.83          |
| LAYERWISE CMA-ES | 15360       | 0              | 1.35s      | 0.6177    | 0.93          |    


## References