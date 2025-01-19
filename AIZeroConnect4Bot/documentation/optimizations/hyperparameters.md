
# Hyperparameter Optimization

Using Bayesian Optimization (available through `Optuna`), the hyperparameters of the model can be optimized (see `opt.py`). Each hyperparameter optimization is relatively expensive, as it requires training the model for multiple epochs. The hyperparameters are optimized based on the validation loss of the model. Choosing the right hyperparameters can greatly improve the performance of the model and the training speed.
