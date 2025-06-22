
# Hyperparameter Optimization

Hyperparameters are crucial for the performance of machine learning models. They are parameters that are set before the training process begins and can significantly affect the model's ability to learn from the data. In this project, we have several hyperparameters that can be tuned to improve the model's performance.

Using Bayesian Optimization (available through `Optuna`), the hyperparameters of the model can be optimized (see `opt.py`). Each hyperparameter optimization is relatively expensive, as it requires training the model for multiple epochs. The hyperparameters are optimized based on the validation loss of the model. Choosing the right hyperparameters can greatly improve the performance of the model and the training speed.

Use of Population-Based Training (PBT) is something that should be explored in future work, as it allows for more dynamic and efficient hyperparameter optimization by maintaining a population of models and evolving them over time.
