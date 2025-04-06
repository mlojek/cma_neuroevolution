# CMA-ES Neuroevolution
This project probes the possibility and feasibility of training neural networks with CMA-ES algorithm.

Artificial neural networks play a crucial role in modern machine learning. These deep learning models are mostly trained using gradient methods, e.g. SGD and ADAM. These methods however, are not the only ways of finding optimal parameter values. Neural networks can be trained using genetic algorithms - such process is called neuroevolution. This project explores the viablity of using CMA-ES algorithm to optimize simple neural networks. CMA-ES is the state-of-the-art black-box optimization which optimizes a function by randomly sampling the search space with a multivariate normal distribution. 

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
Before running the project a configs file needs to be created. Example configs can be found in `configs/` directory. Then, to prepare the dataset run:
```
python prepare_data.py DATASET_CONFIG
```
To train the model, run:
```
python train.py CONFIG
```
Finally, to evaluate a trained model:
```
python evaluate.py CONFIG
```
All config fields are described in `configs/data_model.py` script.

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

| Training mode      | Epochs | Model evaluations | Gradient evaluations | Training time (s) | Val loss | Test loss | Test accuracy |
|--------------------|--------|-------------------|----------------------|-------------------|----------|-----------|---------------|
| SGD                | 200    | 24000             | 1200                 | 2.33              | 0.63     | 0.6220    | 0.93          |
| ADAM               | 40     | 4800              | 240                  | 1.8               | 0.63     | 0.6222    | 0.93          |
| CMA-ES             | 5      | 14100             | 0                    | 4.17              | 0.66     | 0.7039    | 0.83          |
| LAYERWISE CMA-ES   | 8      | 15360             | 0                    | 1.35              | 0.62     | 0.6177    | 0.93          |    


### MNIST dataset

| Training mode      | Epochs | Model evaluations | Gradient evaluations | Training time (s) | Val loss | Test loss | Test accuracy |
|--------------------|--------|-------------------|----------------------|-------------------|----------|-----------|---------------|
| SGD                | 200    | 24000             | 1200                 | 2.33              | 0.63     | 0.6220    | 0.93          |
| ADAM               | 40     | 4800              | 240                  | 1.8               | 0.63     | 0.6222    | 0.93          |
| CMA-ES             | 5      | 14100             | 0                    | 4.17              | 0.66     | 0.7039    | 0.83          |
| LAYERWISE CMA-ES   | 8      | 15360             | 0                    | 1.35              | 0.62     | 0.6177    | 0.93          |    


## References