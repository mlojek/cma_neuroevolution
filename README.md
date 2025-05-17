# CMA-ES Neuroevolution

This project probes the possibility and feasibility of training neural networks with CMA-ES algorithm.

Artificial neural networks play a crucial role in modern machine learning. These deep learning models are mostly trained using gradient methods, e.g. SGD and ADAM. These methods however, are not the only ways of finding optimal parameter values. Neural networks can be trained using genetic algorithms - such process is called neuroevolution. This project explores the viablity of using CMA-ES algorithm to optimize simple neural networks. CMA-ES is the state-of-the-art black-box optimization which optimizes a function by randomly sampling the search space with a multivariate normal distribution.

## Installation

For this project, python 3.13 is recommended. To install dependencies run:

```sh
make install
```

This project uses wandb.ai for experiment tracking. During model training, the train and val average loss and accuracy are logged to the service. To enable wandb.ai in this project, run:

```bash
export WANDB_API_KEY=<your_api_key>
wandb login
```

And enable wandb logging in your experiment config.

## How to run

To tune hyperparameters for a given model run:

```
python -m src.tune_hyperparameters.py [-h] [--trials TRIALS] config

positional arguments:
  config           Path to experiment config JSON file.

options:
  -h, --help       show this help message and exit
  --trials TRIALS  Number of trials to perform.
```

To run an experiment with a given config:

```
python src.experiment.py [-h] [--runs RUNS] config

positional arguments:
  config       Path to JSON config file.

options:
  -h, --help   show this help message and exit
  --runs RUNS  Number of model trainings to perform.
```

To perform a comparative analysis of a directory with experiment results CSV files:

```
python -m src.evaluate_results.py [-h] input_dir

positional arguments:
  input_dir   Directory to read CSV result files from.

options:
  -h, --help  show this help message and exit
```

## Experiments

In this project, SGD and Adam gradient optimizers were compared with layerwise and whole-model CMA-ES optimization in classification task using a simple multi-layer perceptron with two fully connected layers.

SGD is the simplest gradient optimizer, selected as a baseline for the results. Adam is widely considered the best and most versatile gradient optimizer, so it was selected as the state-of-the-art method.

For CMA-ES methods, two approaches were tested. The whole-model approach is the optimization of all of the model parameters using one CMA-ES optimizer. The layerwise approach uses two separate CMA-ES optimizers for each of the model layers. For each training batch it separately optimizes each layer, and then updates them at the same time.

The optimizers are tested on two datasets:

- Iris, a small dataset of 4 continous features signifying the dimensions of the petals of iris flowers. These flowers belong to 3 species. The dataset contains only 150 samples.
- MNIST, a computer vision dataset containing 60'000 images of grayscale handwritten digits from 0 to 9. To reduce the dimensionality of the problem they were downscaled from 28x28 pixels to 14x14.

The primary metric for comparison of the models was the final test dataset accuracy and crossentropy loss. Further analyses were conducted for training and validation losses and accuracies, as well as the training time in seconds and the number of model and gradient evaluations. The training used an early stopping mechanism to prevent overfitting. This mean that the training was stopped after the validation loss value didn't improve by a predefined margin for 3 consecutive epochs.

To ensure that the comparison was fair and that all optimizers worked to the best of their ability, hyperparameter tuning was performed. For gradient optimizers, the learning rate was tuned, while for CMA-ES methods the population size and the starting sigma value were tuned.

Each optimizer was run 25 times, with the seeds shared between the respective runs of each of the optimizers to ensure the same starting weights of the model. The resulting metrics were then compared using Mann Whitney U test to check for statistical significance.

## Results

A comprehensive results analysis was conducted in the [results](results.md) document. In the tables below the results are summarizes, with best results based on statistical tests `highlighted`.

### Iris dataset

| optimizer | time | train_loss | train_acc | val_loss | val_acc | test_loss | test_acc | model_evals |
|-----------------|--------------|--------------|-------------|--------------|-------------|---------------|--------------|---------------|
| sgd | `0.04731` | 0.7 | 0.8542 | 0.7476 | 0.7987 | 0.7343 | 0.8173 | `1450` |
| adam | 0.06589 | `0.5724` | `0.9804` | `0.6334` | `0.924` | `0.6124` | `0.9413` | 1919 |
| cmaes | 10.06 | 0.672 | 0.8782 | 0.6927 | 0.8587 | 0.7125 | 0.8387 | 1.885e+04 |
| layerwise_cmaes | 0.6935 | 0.6318 | 0.9213 | `0.6638` | `0.8867` | 0.6687 | 0.884 | 5.556e+04 |

### MNIST dataset

| optimizer | time | train_loss | train_acc | val_loss | val_acc | test_loss | test_acc | model_evals |
|-----------------|-----------|--------------|-------------|------------|------------|-------------|------------|---------------|
| sgd | 23.23 | 1.574 | 0.8969 | 1.582 | 0.8867 | 1.58 | 0.8887 | `3.355e+06` |
| adam | `20.28` | `1.544` | `0.9266` | `1.562` | `0.905` | `1.559` | `0.9074` | `3.357e+06` |
| cmaes | 1336 | 1.749 | 0.7116 | 1.754 | 0.707 | 1.752 | 0.709 | 6.334e+07 |
| layerwise_cmaes | 1294 | 1.773 | 0.688 | 1.777 | 0.6843 | 1.776 | 0.6851 | 1.056e+08 |

## Conclusions

- Neural network optimization using CMA-ES does not yield better results than using gradient methods.
- CMA-ES becomes very slow for bigger models.
- Using separate CMA-ES optimizer for each layer of the network instead of one optimizer for the whole network works best when the layers have similar numbers of optimized parameters.
- Adam optimizer performed the best out of all the optimizers.
- Adam optimizer also performed the most reliably, having the lowest standard deviations of results.
