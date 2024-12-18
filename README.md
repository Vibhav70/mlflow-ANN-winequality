# README: Wine Quality Prediction using ANN with MLflow and Hyperopt

## Project Overview
This project demonstrates how to train an Artificial Neural Network (ANN) to predict wine quality using the UCI Wine Quality dataset. The focus is on leveraging MLflow for tracking experiments and Hyperopt for hyperparameter tuning. The ANN is trained using the SGD optimizer, where learning rate (`lr`) and momentum are optimized.

## Key Features
1. **Artificial Neural Network (ANN)**: A simple ANN model with two dense layers, implemented using Keras.
2. **MLflow Integration**: Tracks experiments, logs hyperparameters, metrics, and trained models.
3. **Hyperparameter Tuning**: Uses Hyperopt to find the optimal values for learning rate (`lr`) and momentum.

## Requirements
- Python 3.8+
- Required Libraries:
  - TensorFlow
  - MLflow
  - Hyperopt
  - NumPy

Install the dependencies using:
```bash
pip install tensorflow mlflow hyperopt numpy
```

## Dataset
The project uses the UCI Wine Quality dataset, which contains physicochemical attributes of wine samples and their quality ratings.

### Preprocessing Steps
- Normalize the input features.
- Split the data into training, validation, and testing sets.

## Code Structure
### 1. **`train_model` Function**
Defines the ANN architecture, compiles the model, and trains it using the given hyperparameters. MLflow is used to log the parameters, evaluation metrics, and the trained model.

#### Parameters:
- `params`: Dictionary containing `lr` and `momentum` values.
- `epochs`: Number of training epochs.
- `train_x`, `train_y`: Training data.
- `test_x`, `test_y`: Testing data.

#### Logs:
- Hyperparameters (`lr`, `momentum`)
- Evaluation metrics (`eval_rmse`)
- Trained model

### 2. **`objective` Function**
Acts as a wrapper for the `train_model` function to evaluate a single set of hyperparameters.

### 3. **Hyperparameter Search**
The search space for the learning rate (`lr`) and momentum is defined using Hyperopt. The `fmin` function performs optimization by minimizing the evaluation loss (`eval_rmse`).

### 4. **MLflow Experiment**
The code sets up an MLflow experiment and tracks multiple runs. The best hyperparameters and corresponding model are logged.

## Usage
1. **Set Experiment**:
   ```python
   mlflow.set_experiment("/wine-quality")
   ```

2. **Run the Script**:
   Execute the script to start the training and hyperparameter tuning process. The script performs the following:
   - Initializes an MLflow experiment.
   - Conducts hyperparameter optimization using Hyperopt.
   - Logs the best model, hyperparameters, and evaluation metrics to MLflow.

3. **Inspect Results**:
   Open the MLflow UI to view experiment details:
   ```bash
   mlflow ui
   ```
   Access the UI at `http://localhost:5000`.

## Results
- Best Hyperparameters:
  - `lr`: Learning rate (logged in MLflow).
  - `momentum`: Momentum (logged in MLflow).
- Evaluation Metric:
  - `eval_rmse`: Root Mean Squared Error on the validation set.
- Model: The trained model is logged to MLflow.

## Example Output
```bash
Best parameters: {'lr': 0.01, 'momentum': 0.9}
Best eval rmse: 0.1234
```

## Notes
- Ensure that the dataset is preprocessed correctly before running the script.
- Adjust the `max_evals` parameter in Hyperopt for a more exhaustive search.

## Acknowledgements
- UCI Wine Quality Dataset: [https://archive.ics.uci.edu/ml/datasets/wine+quality](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- MLflow Documentation: [https://www.mlflow.org/](https://www.mlflow.org/)
- Hyperopt Documentation: [http://hyperopt.github.io/hyperopt/](http://hyperopt.github.io/hyperopt/)

---
This project showcases the integration of MLflow and Hyperopt for effective model training and hyperparameter tuning.

