# CMA-ES Neuroevolution
This project probes the possibility and feasibility of training neural networks with CMA-ES algorithm.

Artificial neural networks play a crucial role in modern machine learning. These deep learning are mostly trained using gradient methods, e.g. SGD and ADAM. These methods however, are not the only ways of finding optimal parameter values. Neural networks can be trained using genetic algorithms. Such process is called neuroevolution. This project explores the viablity of using CMA-ES algorithm to optimize neural networks. CMA-ES is the state-of-the-art black-box optimization which optimizes a function by randomly sampling the search space with a multivariate normal distribution. 

## Experiments outline
For simplicity's sake, the optimized neural network is a multi-layer perceptron classifier for iris and MNIST datasets. A few strategies are compared:
- Gradient-based:
    - Adam optimizer    
    - Stochastic Gradient Descent
- CMA-based:
    - Whole-model CMA-ES
    - Layerwise CMA-ES

To compare the results, a few metrics are used:
- final loss and accuracy on test set
- Training time in seconds and the number of function evaluations (+gradient calculations).

The models are trained until the loss on validation set is at it's minimum.

## Installation
For this project, python 3.13 is recommended. To install dependencies run:
```
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

## Experiment results
- run training a 10 times
- see on which epoch the val loss minimizes or stalls
- select that number of epochs
- run training 5 times and select the best run 

| Dataset name | Training mode      | Epochs | Model evaluations | Gradient evaluations | Training time (s) | Val loss | Test loss | Test accuracy |
|--------------|--------------------|--------|-------------------|----------------------|-------------------|----------|-----------|---------------|
| Iris         | SGD                | 200    | 24000             | 1200                 | 2.33              | 0.63     | 0.6220    | 0.93          |
| Iris         | ADAM               | 40     | 4800              | 240                  | 1.8               | 0.63     | 0.6222    | 0.93          |
| Iris         | CMA-ES             | 5      | 14100             | 0                    | 4.17              | 0.66     | 0.7039    | 0.83          |
| Iris         | LAYERWISE CMA-ES   | 8      | 15360             | 0                    | 1.35              | 0.62     | 0.6177    | 0.93          |    
| MNIST        | SGD                |
| MNIST        | ADAM               |
| MNIST        | CMA-ES             |
| MNIST        | LAYERWISE CMA-ES   |






## References