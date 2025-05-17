# CMA-ES Neuroevolution - Experiment results

To evaluate the performance of CMA-ES neuroevolution, I conducted a comparative study against gradient optimization methods. Each optimizer was executed 25 independent times to obtain various performance metrics. To ensure fair comparisons, a constant random seed (42) was used for dataset splitting and CMA-ES initialization across all experiments. Model layer initialization was performed with a different random seed for each of the 25 runs (ranging from 0 to 24), applied for all optimizers within a given run. This setup was selected to evaluate the optimizers for various "starting points" for the optimized model, while also ensuring fair comparison between the various methods.

To ensure that each optimizer works to the best of it's ability, hyperparameter optimization was performed using the `src/tune_hyperparameters.py` script. Specifically, the learning rate was optimized for gradient-based methods, and both the population size and initial sigma were optimized for CMA-ES variants.

To prevent overfitting and optimize training efficiency, an early stopping mechanism was implemented. During training, if the validation loss did not improve by a set margin selected for each dataset, for three consecutive epochs, the training process was ended to avoid overfitting.

Performance was assessed based on a comprehensive set of metrics, including loss and accuracy on the training, validation, and testing datasets. Additionally, the training time in seconds and the total number of model and gradient evaluations required for convergence were measured.

To determine the statistical significance of the observed differences, the non-parametric Mann-Whitney U test was selected. The resulting p-values are reported in the subsequent tables. For measurements of loss, training time, and the number of evaluations, lower values indicate better performance, while for accuracy, higher values are better. A p-value below 0.05 signifies that the performance of the optimizer in the row is statistically significantly better than that of the optimizer in the column.

# Iris dataset

Iris dataset is a classification dataset with 150 samples of 4 continous parameters of the dimensions of iris petals. The samples below to 3 classes signifying a different iris species. This dataset is small and low-dimensional, and should be and easy task for the neural network.

### Mean values

The table below lists the mean values of metrics for each optimizer. The `highlighed` values signify the best performance based on the statistical tests.

| optimizer | time | train_loss | train_acc | val_loss | val_acc | test_loss | test_acc | model_evals |
|-----------------|--------------|--------------|-------------|--------------|-------------|---------------|--------------|---------------|
| sgd | `0.04731` | 0.7 | 0.8542 | 0.7476 | 0.7987 | 0.7343 | 0.8173 | `1450` |
| adam | 0.06589 | `0.5724` | `0.9804` | `0.6334` | `0.924` | `0.6124` | `0.9413` | 1919 |
| cmaes | 10.06 | 0.672 | 0.8782 | 0.6927 | 0.8587 | 0.7125 | 0.8387 | 1.885e+04 |
| layerwise_cmaes | 0.6935 | 0.6318 | 0.9213 | `0.6638` | `0.8867` | 0.6687 | 0.884 | 5.556e+04 |

### Training time

- The fastest optimizer was SGD, probably due to it's simplicity.
- Gradient optimizers were significantly faster than CMA-ES methods.
- Layerwise CMA-ES was 14 times faster than whole-model CMA-ES. This might be because of O(n^3) time complexity of CMA-ES metaheuristic, and optimizing two neural network layers separately greatly reduced the overall complexity.

| optimizer | mean | std | sgd | adam | cmaes | layerwise_cmaes |
|-----------------|----------|---------|-------|--------|---------|-------------------|
| `sgd` | 0.04731 | 0.03879 | - | 0.000 | 0.000 | 0.000 |
| adam | 0.06589 | 0.03002 | 1.000 | - | 0.000 | 0.000 |
| cmaes | 10.06 | 3.107 | 1.000 | 1.000 | - | 1.000 |
| layerwise_cmaes | 0.6935 | 0.2064 | 1.000 | 1.000 | 0.000 | - |

### Training loss and accuracy

- The lowest training loss was observed for Adam optimizer, meaning that this optimizer was the best at finding the global minimum of the model.
- CMA-ES methods performed on par with SGD.
- Layerwise CMA-ES performed better than whole-model CMA-ES.

| optimizer | mean | std | sgd | adam | cmaes | layerwise_cmaes |
|-----------------|--------|----------|-------|--------|---------|-------------------|
| sgd | 0.7 | 0.173 | - | 1.000 | 0.094 | 0.751 |
| `adam` | 0.5724 | 0.001306 | 0.000 | - | 0.000 | 0.000 |
| cmaes | 0.672 | 0.06188 | 0.910 | 1.000 | - | 0.999 |
| layerwise_cmaes | 0.6318 | 0.1161 | 0.255 | 1.000 | 0.001 | - |

- Same observations apply to the training accuracy.

| optimizer | mean | std | sgd | adam | cmaes | layerwise_cmaes |
|-----------------|--------|----------|-------|--------|---------|-------------------|
| sgd | 0.8542 | 0.1763 | - | 0.999 | 0.084 | 0.577 |
| `adam` | 0.9804 | 0.004843 | 0.001 | - | 0.000 | 0.000 |
| cmaes | 0.8782 | 0.06224 | 0.919 | 1.000 | - | 1.000 |
| layerwise_cmaes | 0.9213 | 0.1276 | 0.430 | 1.000 | 0.000 | - |

### Validation loss and accuracy

- Adam and layerwise CMA-ES tied for the best performance.
- SGD and whole-model CMA-ES tied for the worst performance.
- Layerwise CMA-ES performed significantly better than whole-model CMA-ES.

| optimizer | mean | std | sgd | adam | cmaes | layerwise_cmaes |
|---------------------|--------|----------|-------|--------|---------|-------------------|
| sgd | 0.7476 | 0.1472 | - | 1.000 | 0.637 | 0.998 |
| `adam` | 0.6334 | 0.006367 | 0.000 | - | 0.000 | 0.546 |
| cmaes | 0.6927 | 0.05911 | 0.371 | 1.000 | - | 0.998 |
| `layerwise_cmaes` | 0.6638 | 0.1092 | 0.002 | 0.461 | 0.002 | - |

- Same observations apply to the validation accuracy.

| optimizer | mean | std | sgd | adam | cmaes | layerwise_cmaes |
|---------------------|--------|---------|-------|--------|---------|-------------------|
| sgd | 0.7987 | 0.1473 | - | 1.000 | 0.829 | 1.000 |
| `adam` | 0.924 | 0.01528 | 0.000 | - | 0.000 | 0.171 |
| cmaes | 0.8587 | 0.06257 | 0.176 | 1.000 | - | 0.999 |
| `layerwise_cmaes` | 0.8867 | 0.124 | 0.000 | 0.834 | 0.002 | - |

### Test loss and accuracy

- Adam outperformed all other methods.
- Layerwise CMA-ES placed second, SGD third and whole-model CMA-ES last.
- Layerwise CMA-ES once again outperformed whole-model CMA-ES.

| optimizer | mean | std | sgd | adam | cmaes | layerwise_cmaes |
|-----------------|--------|----------|-------|--------|---------|-------------------|
| sgd | 0.7343 | 0.1533 | - | 1.000 | 0.249 | 0.954 |
| `adam` | 0.6124 | 0.005971 | 0.000 | - | 0.000 | 0.000 |
| cmaes | 0.7125 | 0.07849 | 0.758 | 1.000 | - | 0.996 |
| layerwise_cmaes | 0.6687 | 0.1131 | 0.048 | 1.000 | 0.004 | - |

- Same observations apply to the test accuracy.

| optimizer | mean | std | sgd | adam | cmaes | layerwise_cmaes |
|-----------------|--------|---------|-------|--------|---------|-------------------|
| sgd | 0.8173 | 0.1558 | - | 1.000 | 0.305 | 0.963 |
| `adam` | 0.9413 | 0.01453 | 0.000 | - | 0.000 | 0.002 |
| cmaes | 0.8387 | 0.07916 | 0.702 | 1.000 | - | 0.999 |
| layerwise_cmaes | 0.884 | 0.1266 | 0.039 | 0.998 | 0.001 | - |

### Model and gradient evaluations

- SGD performed the least model and gradient evaluations.
- Adam placed second, whole-model CMA-ES third and layerwise CMA-ES last.
- Gradient methods significantly outperformed CMA-ES methods.

| optimizer | mean | std | sgd | adam | cmaes | layerwise_cmaes |
|-----------------|--------------|--------------|-------|--------|---------|-------------------|
| `sgd` | 1450 | 520.4 | - | 0.001 | 0.000 | 0.000 |
| adam | 1919 | 434.2 | 0.999 | - | 0.000 | 0.000 |
| cmaes | 1.885e+04 | 4971 | 1.000 | 1.000 | - | 0.000 |
| layerwise_cmaes | 5.556e+04 | 1.224e+04 | 1.000 | 1.000 | 1.000 | - |

## Conclusions

- Adam optimizer performs best out of all four optimizers in terms of the final model accuracy.
- Adam also performs best in metrics of loss value and accuracy.
- In metrics of loss and accuracy, Adam has the lowest standard deviation values, meaning that regardless of the starting weights of the model it always performs well. Whole-model CMA-ES was second, while SGD and layerwise CMA-ES had the highest standard deviations.
- Layerwise CMA-ES performed substantially better than whole-model CMA-ES in all metrics except for the number of model evaluations.
- Despite layerwise CMA-ES requiring 4x more model evaluations, it learned 14x faster than whole-model CMA-ES. The use of two CMA-ES optimizers instead of one for the whole model reduced the time complexity of CMA-ES methaheurisitic.
- Layerwise CMA-ES beat SGD on test set. Whole model CMA-ES performed on par with SGD.

# MNIST dataset

MNIST is a computer vision classification dataset consisting images of 28x28 grayscale handwritten digits from 10 classes. The training split of the dataset, used in this project, consists of 60'000 such images. To reduce the model complexity, the images were downscaled to 14x14 pixels.

### Mean values

The table below lists the mean values of metrics for each optimizer. The `bold` values signify the best performance based on the statistical tests.

| optimizer | time | train_loss | train_acc | val_loss | val_acc | test_loss | test_acc | model_evals |
|-----------------|-----------|--------------|-------------|------------|------------|-------------|------------|---------------|
| sgd | 23.23 | 1.574 | 0.8969 | 1.582 | 0.8867 | 1.58 | 0.8887 | `3.355e+06` |
| adam | `20.28` | `1.544` | `0.9266` | `1.562` | `0.905` | `1.559` | `0.9074` | `3.357e+06` |
| cmaes | 1336 | 1.749 | 0.7116 | 1.754 | 0.707 | 1.752 | 0.709 | 6.334e+07 |
| layerwise_cmaes | 1294 | 1.773 | 0.688 | 1.777 | 0.6843 | 1.776 | 0.6851 | 1.056e+08 |

### Training time

- Gradient methods take 50-70 times quicker to train the model than CMA-ES methods.
- Adam learned the quickest.
- There's no significant difference in training time between layerwise and whole-model CMA-ES, even though there was a big difference for Iris dataset. This might be because in this model the two layers of the network differed in the number of parameters, contrary to Iris dataset where the layers were of similar size. There's also more data in this dataset, so the time to evaluate the model is longer than it was for Iris dataset, and so the difference in time needed to generate new solutions by CMA-ES does not influence the total training time that much.

| optimizer | mean | std | sgd | adam | cmaes | layerwise_cmaes |
|-----------------|---------|---------|-------|--------|---------|-------------------|
| sgd | 23.23 | 6.198 | - | 0.967 | 0.000 | 0.000 |
| `adam` | 20.28 | 5.336 | 0.034 | - | 0.000 | 0.000 |
| cmaes | 1336 | 475.9 | 1.000 | 1.000 | - | 0.539 |
| layerwise_cmaes | 1294 | 371.3 | 1.000 | 1.000 | 0.469 | - |

### Training loss and accuracy

- Adam significantly beat all other methods.
- Both gradient optimizers performed better than CMA-ES methods.
- Both CMA-ES methods performed comparably well.

| optimizer | mean | std | sgd | adam | cmaes | layerwise_cmaes |
|-----------------|--------|---------|-------|--------|---------|-------------------|
| sgd | 1.574 | 0.04179 | - | 1.000 | 0.000 | 0.000 |
| `adam` | 1.544 | 0.02932 | 0.000 | - | 0.000 | 0.000 |
| cmaes | 1.749 | 0.08144 | 1.000 | 1.000 | - | 0.274 |
| layerwise_cmaes | 1.773 | 0.08966 | 1.000 | 1.000 | 0.733 | - |

- Same observations apply to the training accuracy.

| optimizer | mean | std | sgd | adam | cmaes | layerwise_cmaes |
|-----------------|--------|---------|-------|--------|---------|-------------------|
| sgd | 0.8969 | 0.04367 | - | 1.000 | 0.000 | 0.000 |
| `adam` | 0.9266 | 0.03064 | 0.000 | - | 0.000 | 0.000 |
| cmaes | 0.7116 | 0.08164 | 1.000 | 1.000 | - | 0.280 |
| layerwise_cmaes | 0.688 | 0.08965 | 1.000 | 1.000 | 0.726 | - |

### Validation loss and accuracy

- Adam significantly outperformed all other methods.
- Both gradient optimizers significantly outperformed CMA-ES methods.
- There's no significant difference in performance for both CMA-ES methods.

| optimizer | mean | std | sgd | adam | cmaes | layerwise_cmaes |
|-----------------|--------|---------|-------|--------|---------|-------------------|
| sgd | 1.582 | 0.04109 | - | 0.999 | 0.000 | 0.000 |
| `adam` | 1.562 | 0.02798 | 0.001 | - | 0.000 | 0.000 |
| cmaes | 1.754 | 0.0813 | 1.000 | 1.000 | - | 0.280 |
| layerwise_cmaes | 1.777 | 0.08803 | 1.000 | 1.000 | 0.726 | - |

- Same observations apply to the validation accuracy.

| optimizer | mean | std | sgd | adam | cmaes | layerwise_cmaes |
|-----------------|--------|---------|-------|--------|---------|-------------------|
| sgd | 0.8867 | 0.04291 | - | 0.993 | 0.000 | 0.000 |
| `adam` | 0.905 | 0.02914 | 0.008 | - | 0.000 | 0.000 |
| cmaes | 0.707 | 0.0815 | 1.000 | 1.000 | - | 0.280 |
| layerwise_cmaes | 0.6843 | 0.08802 | 1.000 | 1.000 | 0.726 | - |

### Test loss and accuracy

- Adam significantly outperformed all other methods.
- Both gradient optimizers outperformed CMA-ES methods.
- There's no significant difference in performance for both CMA-ES methods.

| optimizer | mean | std | sgd | adam | cmaes | layerwise_cmaes |
|-----------------|--------|---------|-------|--------|---------|-------------------|
| sgd | 1.58 | 0.04137 | - | 1.000 | 0.000 | 0.000 |
| `adam` | 1.559 | 0.0289 | 0.000 | - | 0.000 | 0.000 |
| cmaes | 1.752 | 0.08075 | 1.000 | 1.000 | - | 0.280 |
| layerwise_cmaes | 1.776 | 0.08903 | 1.000 | 1.000 | 0.726 | - |

- Same observations apply to the test accuracy.

| optimizer | mean | std | sgd | adam | cmaes | layerwise_cmaes |
|-----------------|--------|---------|-------|--------|---------|-------------------|
| sgd | 0.8887 | 0.0432 | - | 0.995 | 0.000 | 0.000 |
| `adam` | 0.9074 | 0.02988 | 0.005 | - | 0.000 | 0.000 |
| cmaes | 0.709 | 0.08089 | 1.000 | 1.000 | - | 0.287 |
| layerwise_cmaes | 0.6851 | 0.08902 | 1.000 | 1.000 | 0.720 | - |

### Model and gradient evaluations

- Both graident methods performed comparable numbers of model evaluations.
- Gradient methods performed significantly less evaluations than CMA-ES methods.
- Whole-model CMA-ES performed significantly less model evaluations than layerwise CMA-ES.

| optimizer | mean | std | sgd | adam | cmaes | layerwise_cmaes |
|-----------------|-----------|-----------|-------|--------|---------|-------------------|
| `sgd` | 3.355e+06 | 7.789e+05 | - | 0.600 | 0.000 | 0.000 |
| `adam` | 3.357e+06 | 7.623e+05 | 0.408 | - | 0.000 | 0.000 |
| cmaes | 6.334e+07 | 2.179e+07 | 1.000 | 1.000 | - | 0.000 |
| layerwise_cmaes | 1.056e+08 | 3.008e+07 | 1.000 | 1.000 | 1.000 | - |

## Conclusions

- Adam optimizer outperformed all other optimizers.
- Standard deviations are lowest for Adam optimizer.
- The differences in standard deviations were not as big as in the Iris dataset.
- Gradient methods outperformed both CMA-ES methods.
- There were no significant differences in loss and accuracy between the CMA-ES methods. This might be because the numbers of paramaters in layers in MNIST model (1970, 110) were not as balanced as in Iris dataset (50, 33).
- The differences in training time between gradient methods and CMA-ES methods were substantial, signifying that CMA-ES methods do not scale well for bigger models.

# General conclusions

- Neural network optimization using CMA-ES does not yield better results than using gradient methods.
- CMA-ES becomes very slow for bigger models.
- Using separate CMA-ES optimizer for each layer of the network instead of one optimizer for the whole network works best when the layers have similar numbers of optimized parameters.
- Adam optimizer performed the best out of all the optimizers.
- Adam optimizer also performed the most reliably, having the lowest standard deviations of results.
