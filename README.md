# CMA-ES Neuroevolution
Neural networks are generally trained using gradient descent methods. This method however, is not the only way to optimize a function. CMA-ES is the state-of-the-art black-box optimization which optimizes a function by randomly sampling the search space with a multivariate normal distribution.

This project explores the viablity of using CMA-ES optimizer to optimize neural networks. Such approach is known as Neuroevolution.

## Experiments outline
For simplicity's sake, the optimized neural network is a multi-layer perceptron classifier for iris dataset. A few strategies are compared:
- Gradient-based:
    - Adam optimizer    
    - Stochastic Gradient Descent
- CMA-based:
    - Whole-model CMA-ES
    - Simultaneous layerwise CMA-ES
    - Front-to-back layerwise CMA-ES
    - Back-to-front layerwise CMA-ES

To compare the results, a few metrics are used:
- final loss and accuracy on test set
- Training time in the number of function evaluations and seconds
- Resource consumption (CPU)

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

## References