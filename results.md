# CMA-ES Neuroevolution - Detailed results

TODO setup - 25 runs, each with seed that is the number of run.
TODO why those metrics
TODO what do all tables mean (pvales of row < column or > in case of accuracy)
## Results - Iris dataset
### Mean values
| Optimizer                                        | Model evals | Gradient evals | Train time (s) | Train loss | Train accuracy | Test loss | Test accuracy |
|--------------------------------------------------|-------------|----------------|----------------|------------|----------------|-----------|---------------|
| [SGD](configs/iris_sgd.json)                     | **1830**        | **84**             | **0.1844**         |            |                | 0.6145    | 0.9333        |
| [ADAM](configs/iris_adam.json)                   | 1350        | 60             | 0.1804         |            |                | 0.6033    | 0.9333        |
| [CMA-ES](configs/iris_cmaes.json)                | 22410       | n/a            | 10.858         |            |                | 0.6607    | 0.9           |
| [LAYERWISE CMA-ES](configs/iris_layerwise.json)  | 50130       | n/a            | 0.7341         |            |                | 0.6542    | 0.9           |  

### Model evaluations
| Optimizer | Median    | IQR   | SGD   | ADAM  | CMA-ES | LAYERWISE |
|-----------|-----------|-------|-------|-------|--------|-----------|
| SGD       |           |       | -     |       |        |           |
| ADAM      |           |       |       | -     |        |           |
| CMA-ES    |           |       |       |       | -      |           |
| LAYERWISE |           |       |       |       |        | -         |

### Gradient evaluations
| Optimizer | Median    | IQR   | SGD   | ADAM  | CMA-ES | LAYERWISE |
|-----------|-----------|-------|-------|-------|--------|-----------|
| SGD       |           |       | -     |       |        |           |
| ADAM      |           |       |       | -     |        |           |
| CMA-ES    |           |       |       |       | -      |           |
| LAYERWISE |           |       |       |       |        | -         |

### Training time
| Optimizer | Median    | IQR   | SGD   | ADAM  | CMA-ES | LAYERWISE |
|-----------|-----------|-------|-------|-------|--------|-----------|
| SGD       |           |       | -     |       |        |           |
| ADAM      |           |       |       | -     |        |           |
| CMA-ES    |           |       |       |       | -      |           |
| LAYERWISE |           |       |       |       |        | -         |

### Training loss
| Optimizer | Median    | IQR   | SGD   | ADAM  | CMA-ES | LAYERWISE |
|-----------|-----------|-------|-------|-------|--------|-----------|
| SGD       |           |       | -     |       |        |           |
| ADAM      |           |       |       | -     |        |           |
| CMA-ES    |           |       |       |       | -      |           |
| LAYERWISE |           |       |       |       |        | -         |

### Training accuracy
| Optimizer | Median    | IQR   | SGD   | ADAM  | CMA-ES | LAYERWISE |
|-----------|-----------|-------|-------|-------|--------|-----------|
| SGD       |           |       | -     |       |        |           |
| ADAM      |           |       |       | -     |        |           |
| CMA-ES    |           |       |       |       | -      |           |
| LAYERWISE |           |       |       |       |        | -         |

### Test loss
| Optimizer | Median    | IQR   | SGD   | ADAM  | CMA-ES | LAYERWISE |
|-----------|-----------|-------|-------|-------|--------|-----------|
| SGD       |           |       | -     |       |        |           |
| ADAM      |           |       |       | -     |        |           |
| CMA-ES    |           |       |       |       | -      |           |
| LAYERWISE |           |       |       |       |        | -         |

### Test accuracy
| Optimizer | Median    | IQR   | SGD   | ADAM  | CMA-ES | LAYERWISE |
|-----------|-----------|-------|-------|-------|--------|-----------|
| SGD       |           |       | -     |       |        |           |
| ADAM      |           |       |       | -     |        |           |
| CMA-ES    |           |       |       |       | -      |           |
| LAYERWISE |           |       |       |       |        | -         |

TODO train loss and train accuracy
TODO eliminate reproducibility and run 25 times
TODO every of the 25 runs each time should have the same weights - seed torch weight initialization with run index?


### MNIST dataset
| Training mode                                    | Model evals | Gradient evals | Train time (s) | Test loss | Test accuracy |
|--------------------------------------------------|-------------|----------------|----------------|-----------|---------------|
| [SGD](configs/mnist_sgd.json)                    | 2.89e6      | 1062           | 18.89          | 1.5420    | 0.9251        |
| [ADAM](configs/mnist_adam.json)                  | 2.99e6      | 1098           | 21.91          | 1.5365    | 0.9275        |
| [CMA-ES](configs/mnist_cmaes.json)               | 8.88e7      | n/a            | 4680           | 1.6741    | 0.7872        |
| [LAYERWISE CMA-ES](configs/mnist_layerwise.json) | 7.01e7      | n/a            | 813.5          | 1.7581    | 0.7028        |    

