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

## Experiments
To compare the gradient optimizers with CMA-ES, the following experiment setup will be used:
- Datasets: Iris, MNIST (downscaled to 14x14).
- Gradient optimizers: SGD, Adam.
- CMA-ES strategies: one optimizer for all model parameters, one optimizer per model layer.
- Performance metrics: crossentropy loss, accuracy on test set.
- Efficiency metrics: training time in seconds, model evaluations, gradient calculations.

Experiments are performed by running training and testing loop 25 times and averaging the results to achieve statistical significance of the results. This is done in `experiment.py` script, which you can run by:
```bash
python experiment.py path/to/config [--runs NUM_RUNS]
```
All experiment parameters except for optimizer parameters stay the same for each dataset to ensure comparability of the results from different optimizers. To avoid overfitting, early stopping is used to halt the training after no significant improvement in loss value is observed for 3 consecutive epochs. Mean and standard deviations of the values are then used for comparison. Wherever it's possible, random seed is set to allow reproducibility.

To ensure that each optimizer works with it's full potential, hyperparameter tuning is performed using `optuna`. Optuna uses CMA-ES optimizer to find the best hyperparameters to minimize loss function value. For gradient optimizers the learning rate is tuned, while for cma optimizers population size and sigma0 parameters are tuned. Tuning is performed with `tune_hyperparameters` script, which you can run by:
```bash
python tune_hyperparameters.py path/to/config
```

## Results
### Iris dataset
| Training mode                                     | Model evals | Gradient evals | Train time (s) | Test loss | Test accuracy |
|---------------------------------------------------|-------------|----------------|----------------|-----------|---------------|
| [SGD](configs/iris_sgd.json)                      | 1823        | 84             | 0.1959         | 0.5931    | 0.9666        |
| [ADAM](configs/iris_adam.json)                    | 1230        | 54             | 0.1825         | 0.5915    | 0.9666        |
| [CMA-ES](configs/iris_cmaes.json)                 | 20520       | n/a            | 9.8605         | 0.5514    | 1.0           |
| [LAYERWISE CMA-ES](configs/iris_layerwise.json)   | 63990       | n/a            | 0.6379         | 0.5514    | 1.0           |    


### MNIST dataset
| Training mode                                    | Model evals | Gradient evals | Train time (s) | Test loss | Test accuracy |
|--------------------------------------------------|-------------|----------------|----------------|-----------|---------------|
| [SGD](configs/mnist_sgd.json)                    | 2.89e6      | 1062           | 18.89          | 1.5420    | 0.9251        |
| [ADAM](configs/mnist_adam.json)                  | 2.99e6      | 1098           | 21.91          | 1.5365    | 0.9275        |
| [CMA-ES](configs/mnist_cmaes.json)               |             |                |                |           |               |
| [LAYERWISE CMA-ES](configs/mnist_layerwise.json) |             |                |                |           |               |    


## References