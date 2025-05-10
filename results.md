# CMA-ES Neuroevolution - Experiment results
To evaluate the performance of CMA-ES neuroevolution, I conducted a comparative study against gradient optimization methods. Each optimizer was executed 25 independent times to obtain robust performance metrics. To ensure fair comparisons, a consistent random seed (42) was utilized for dataset splitting and CMA-ES initialization across all experiments. Model layer initialization was performed with a different random seed for each of the 25 runs (ranging from 0 to 24), applied for all optimizers within a given run.

To ensure that each optimizer works to the best of it's ability, hyperparameter optimization was performed using the `src/tune_hyperparameters.py` script. Specifically, the learning rate was optimized for gradient-based methods, and both the population size and initial sigma were optimized for CMA-ES variants.

To prevent overfitting and optimize training efficiency, an early stopping mechanism was implemented. Model training was monitored based on the validation loss. If the validation loss did not improve by a predefined margin selected for each dataset, for three consecutive epochs, the training process was ended to avoid overfitting.

Performance was assessed based on a comprehensive set of metrics, including loss and accuracy on the training, validation, and testing datasets. Additionally, the training time in seconds and the total number of model and gradient evaluations required for convergence were measured.

To determine the statistical significance of the observed differences, the non-parametric Mann-Whitney U test was selected. The resulting p-values are reported in the subsequent tables. For measurements of loss, training time, and the number of evaluations, lower values indicate better performance, while for accuracy, higher values are better. A p-value below 0.05 signifies that the performance of the optimizer in the row is statistically significantly better than that of the optimizer in the column.

# Iris dataset
- add problem description
### Mean values
|    | optimizer       |     time |   train_loss |   train_acc |   val_loss |   val_acc |   test_loss |   test_acc |   model_evals |   grad_evals |
|----|-----------------|----------|--------------|-------------|------------|-----------|-------------|------------|---------------|--------------|
|  0 | sgd             |**0.203** |       0.6343 |      0.9187 |     0.6825 |    0.868  |      0.6729 |     0.8787 |  **1753**     |        80.16 |
|  1 | adam            |  0.06589 |   **0.5724** |  **0.9804** | **0.6334** |**0.924**  |  **0.6124** | **0.9413** |  **1835**     |        84.24 |
|  2 | cmaes           | 10.06    |       0.672  |      0.8782 |     0.6927 |    0.8587 |      0.7125 |     0.8387 |     1.885e+04 |         0    |
|  3 | layerwise_cmaes |  0.6935  |       0.6318 |      0.9213 |     0.6638 |    0.8867 |      0.6687 |     0.884  |     5.556e+04 |         0    | 

### Training time
|    | optimizer       |     mean |     std | sgd   | adam   | cmaes   | layerwise_cmaes   |
|----|-----------------|----------|---------|-------|--------|---------|-------------------|
|  0 | **sgd**         |  0.203   | 0.5281  | -     | 0.018  | 0.000   | 0.000             |
|  1 | adam            |  0.06589 | 0.03002 | 0.983 | -      | 0.000   | 0.000             |
|  2 | cmaes           | 10.06    | 3.107   | 1.000 | 1.000  | -       | 1.000             |
|  3 | layerwise_cmaes |  0.6935  | 0.2064  | 1.000 | 1.000  | 0.000   | -                 | 

### Training loss
|    | optimizer       |   mean |      std | sgd   | adam   | cmaes   | layerwise_cmaes   |
|----|-----------------|--------|----------|-------|--------|---------|-------------------|
|  0 | sgd             | 0.6343 | 0.1105   | -     | 1.000  | 0.001   | 0.122             |
|  1 | **adam**        | 0.5724 | 0.001306 | 0.000 | -      | 0.000   | 0.000             |
|  2 | cmaes           | 0.672  | 0.06188  | 1.000 | 1.000  | -       | 0.999             |
|  3 | layerwise_cmaes | 0.6318 | 0.1161   | 0.882 | 1.000  | 0.001   | -                 | 

### Training accuracy
|    | optimizer       |   mean |      std | sgd   | adam   | cmaes   | layerwise_cmaes   |
|----|-----------------|--------|----------|-------|--------|---------|-------------------|
|  0 | sgd             | 0.9187 | 0.1143   | -     | 1.000  | 0.000   | 0.090             |
|  1 | **adam**        | 0.9804 | 0.004843 | 0.001 | -      | 0.000   | 0.000             |
|  2 | cmaes           | 0.8782 | 0.06224  | 1.000 | 1.000  | -       | 1.000             |
|  3 | layerwise_cmaes | 0.9213 | 0.1276   | 0.913 | 1.000  | 0.000   | -                 | 

### Validation loss
|    | optimizer           |   mean |      std | sgd   | adam   | cmaes   | layerwise_cmaes   |
|----|---------------------|--------|----------|-------|--------|---------|-------------------|
|  0 | sgd                 | 0.6825 | 0.08794  | -     | 1.000  | 0.037   | 0.969             |
|  1 | **adam**            | 0.6334 | 0.006367 | 0.000 | -      | 0.000   | 0.546             |
|  2 | cmaes               | 0.6927 | 0.05911  | 0.964 | 1.000  | -       | 0.998             |
|  3 | **layerwise_cmaes** | 0.6638 | 0.1092   | 0.033 | 0.461  | 0.002   | -                 | 

### Validation accuracy
|    | optimizer       |   mean |     std | sgd   | adam   | cmaes   | layerwise_cmaes   |
|----|-----------------|--------|---------|-------|--------|---------|-------------------|
|  0 | sgd             | 0.868  | 0.09254 | -     | 0.999  | 0.056   | 0.948             |
|  1 | **adam**        | 0.924  | 0.01528 | 0.001 | -      | 0.000   | 0.171             |
|  2 | cmaes           | 0.8587 | 0.06257 | 0.946 | 1.000  | -       | 0.999             |
|  3 | layerwise_cmaes | 0.8867 | 0.124   | 0.054 | 0.834  | 0.002   | -                 | 

### Test loss
|    | optimizer       |   mean |      std | sgd   | adam   | cmaes   | layerwise_cmaes   |
|----|-----------------|--------|----------|-------|--------|---------|-------------------|
|  0 | sgd             | 0.6729 | 0.09603  | -     | 1.000  | 0.005   | 0.508             |
|  1 | **adam**        | 0.6124 | 0.005971 | 0.000 | -      | 0.000   | 0.000             |
|  2 | cmaes           | 0.7125 | 0.07849  | 0.995 | 1.000  | -       | 0.996             |
|  3 | layerwise_cmaes | 0.6687 | 0.1131   | 0.500 | 1.000  | 0.004   | -                 | 

### Test accuracy
|    | optimizer       |   mean |     std | sgd   | adam   | cmaes   | layerwise_cmaes   |
|----|-----------------|--------|---------|-------|--------|---------|-------------------|
|  0 | sgd             | 0.8787 | 0.09994 | -     | 0.999  | 0.006   | 0.625             |
|  1 | **adam**        | 0.9413 | 0.01453 | 0.001 | -      | 0.000   | 0.002             |
|  2 | cmaes           | 0.8387 | 0.07916 | 0.994 | 1.000  | -       | 0.999             |
|  3 | layerwise_cmaes | 0.884  | 0.1266  | 0.383 | 0.998  | 0.001   | -                 | 

### Model evaluations
|    | optimizer       |         mean |          std | sgd   | adam   | cmaes   | layerwise_cmaes   |
|----|-----------------|--------------|--------------|-------|--------|---------|-------------------|
|  0 | **sgd**         | 1753         |  621.5       | -     | 0.289  | 0.000   | 0.000             |
|  1 | **adam**        | 1835         |  413.5       | 0.717 | -      | 0.000   | 0.000             |
|  2 | cmaes           |    1.885e+04 | 4971         | 1.000 | 1.000  | -       | 0.000             |
|  3 | layerwise_cmaes |    5.556e+04 |    1.224e+04 | 1.000 | 1.000  | 1.000   | -                 | 

### Gradient evaluations
|    | optimizer       |   mean |   std | sgd   | adam   | cmaes   | layerwise_cmaes   |
|----|-----------------|--------|-------|-------|--------|---------|-------------------|
|  0 | sgd             |  80.16 | 31.07 | -     | 0.289  | 1.000   | 1.000             |
|  1 | adam            |  84.24 | 20.67 | 0.717 | -      | 1.000   | 1.000             |
|  2 | cmaes           |   0    |  0    | 0.000 | 0.000  | -       | 1.000             |
|  3 | layerwise_cmaes |   0    |  0    | 0.000 | 0.000  | 1.000   | -                 | 

## Conclusions
- Adam best
- gradient methods better than cmaes methods in all metrics
- however cmaes based methods still satisfactory
- layerwise better in all metrics than cmaes except for model evaluations, most notable difference in optimization time


# MNIST dataset
- add description of problem

###  Mean values
| optimizer       |     time  |   train_loss |   train_acc |   val_loss |   val_acc |   test_loss |   test_acc |   model_evals | grad_evals |
|-----------------|-----------|--------------|-------------|------------|-----------|-------------|------------|---------------|------------|
| sgd             |    23.23  |        1.574 |      0.8969 |      1.582 |    0.8867 |       1.58  |     0.8887 | **3.355e+06** |      617.8 |
| **adam**        | **20.28** |    **1.544** |  **0.9266** |  **1.562** | **0.905** |   **1.559** | **0.9074** | **3.357e+06** |      618.1 |
| cmaes           |  1336     |        1.749 |      0.7116 |      1.754 |    0.707  |       1.752 |     0.709  |     6.334e+07 |        0   |
| layerwise_cmaes |  1294     |        1.773 |      0.688  |      1.777 |    0.6843 |       1.776 |     0.6851 |     1.056e+08 |        0   | 

###  Training time
There's a substantial difference in training times using gradient and CMA-ES methods. The latter report values around 60-70 times higher, which is noticeable during the training. Adam optimizer trained the model the quickest, performing significantly better than all other methods. Both CMA-ES methods performed statistically similar.
| optimizer       |    mean |     std | sgd   | adam   | cmaes   | layerwise_cmaes   |
|-----------------|---------|---------|-------|--------|---------|-------------------|
| sgd             |   23.23 |   6.198 | -     | 0.967  | 0.000   | 0.000             |
| **adam**        |   20.28 |   5.336 | 0.034 | -      | 0.000   | 0.000             |
| cmaes           | 1336    | 475.9   | 1.000 | 1.000  | -       | 0.539             |
| layerwise_cmaes | 1294    | 371.3   | 1.000 | 1.000  | 0.469   | -                 | 

###  Training loss
Adam optimizer again perfomed substantially better than all other methods. CMA-ES methods performed significantly worse than gradient optimizers. CM
| optimizer       |   mean |     std | sgd   | adam   | cmaes   | layerwise_cmaes   |
|-----------------|--------|---------|-------|--------|---------|-------------------|
| sgd             |  1.574 | 0.04179 | -     | 1.000  | 0.000   | 0.000             |
| **adam**        |  1.544 | 0.02932 | 0.000 | -      | 0.000   | 0.000             |
| cmaes           |  1.749 | 0.08144 | 1.000 | 1.000  | -       | 0.274             |
| layerwise_cmaes |  1.773 | 0.08966 | 1.000 | 1.000  | 0.733   | -                 | 

###  Training accuracy
Adam optimizer again performed the best of all methods.  
| optimizer       |   mean |     std | sgd   | adam   | cmaes   | layerwise_cmaes   |
|-----------------|--------|---------|-------|--------|---------|-------------------|
| sgd             | 0.8969 | 0.04367 | -     | 1.000  | 0.000   | 0.000             |
| **adam**        | 0.9266 | 0.03064 | 0.000 | -      | 0.000   | 0.000             |
| cmaes           | 0.7116 | 0.08164 | 1.000 | 1.000  | -       | 0.280             |
| layerwise_cmaes | 0.688  | 0.08965 | 1.000 | 1.000  | 0.726   | -                 | 

###  Validation loss
| optimizer       |   mean |     std | sgd   | adam   | cmaes   | layerwise_cmaes   |
|-----------------|--------|---------|-------|--------|---------|-------------------|
| sgd             |  1.582 | 0.04109 | -     | 0.999  | 0.000   | 0.000             |
| **adam**        |  1.562 | 0.02798 | 0.001 | -      | 0.000   | 0.000             |
| cmaes           |  1.754 | 0.0813  | 1.000 | 1.000  | -       | 0.280             |
| layerwise_cmaes |  1.777 | 0.08803 | 1.000 | 1.000  | 0.726   | -                 | 

###  Validation accuracy
| optimizer       |   mean |     std | sgd   | adam   | cmaes   | layerwise_cmaes   |
|-----------------|--------|---------|-------|--------|---------|-------------------|
| sgd             | 0.8867 | 0.04291 | -     | 0.993  | 0.000   | 0.000             |
| **adam**        | 0.905  | 0.02914 | 0.008 | -      | 0.000   | 0.000             |
| cmaes           | 0.707  | 0.0815  | 1.000 | 1.000  | -       | 0.280             |
| layerwise_cmaes | 0.6843 | 0.08802 | 1.000 | 1.000  | 0.726   | -                 | 

###  Test loss
| optimizer       |   mean |     std | sgd   | adam   | cmaes   | layerwise_cmaes   |
|-----------------|--------|---------|-------|--------|---------|-------------------|
| sgd             |  1.58  | 0.04137 | -     | 1.000  | 0.000   | 0.000             |
| **adam**        |  1.559 | 0.0289  | 0.000 | -      | 0.000   | 0.000             |
| cmaes           |  1.752 | 0.08075 | 1.000 | 1.000  | -       | 0.280             |
| layerwise_cmaes |  1.776 | 0.08903 | 1.000 | 1.000  | 0.726   | -                 | 

###  Test accuracy
| optimizer       |   mean |     std | sgd   | adam   | cmaes   | layerwise_cmaes   |
|-----------------|--------|---------|-------|--------|---------|-------------------|
| sgd             | 0.8887 | 0.0432  | -     | 0.995  | 0.000   | 0.000             |
| **adam**        | 0.9074 | 0.02988 | 0.005 | -      | 0.000   | 0.000             |
| cmaes           | 0.709  | 0.08089 | 1.000 | 1.000  | -       | 0.287             |
| layerwise_cmaes | 0.6851 | 0.08902 | 1.000 | 1.000  | 0.720   | -                 | 

###  Model evaluations
| optimizer       |      mean |       std | sgd   | adam   | cmaes   | layerwise_cmaes   |
|-----------------|-----------|-----------|-------|--------|---------|-------------------|
| **sgd**         | 3.355e+06 | 7.788e+05 | -     | 0.600  | 0.000   | 0.000             |
| **adam**        | 3.357e+06 | 7.622e+05 | 0.408 | -      | 0.000   | 0.000             |
| cmaes           | 6.334e+07 | 2.179e+07 | 1.000 | 1.000  | -       | 0.000             |
| layerwise_cmaes | 1.056e+08 | 3.008e+07 | 1.000 | 1.000  | 1.000   | -                 | 

###  Gradient evaluations
| optimizer       |   mean |   std | sgd   | adam   | cmaes   | layerwise_cmaes   |
|-----------------|--------|-------|-------|--------|---------|-------------------|
| sgd             |  617.8 | 146   | -     | 0.600  | 1.000   | 1.000             |
| adam            |  618.1 | 142.9 | 0.408 | -      | 1.000   | 1.000             |
| cmaes           |    0   |   0   | 0.000 | 0.000  | -       | 1.000             |
| layerwise_cmaes |    0   |   0   | 0.000 | 0.000  | 1.000   | -                 | 

## Conclusions
In all metrics except for model and gradient evaluations, the Adam optimizer perform significantly better than all other optimization methods.

Both Adam and SGD gradient optimization methods performed significantly better than CMA-ES methods in all metrics (except gradient evaluations, which CMA-ES methods do not perform).

The only statistically significant difference between whole-model and layerwise CMA-ES has been observed in the number of model evaluations, in which whole-model cmaes performed less than layerwise cmaes.
