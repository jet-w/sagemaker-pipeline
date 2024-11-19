import argparse
import joblib
import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters for the SVM model
    #parser.add_argument('--C', type=float, default=1.0, help='Regularization parameter')
    #parser.add_argument('--kernel', type=str, default='linear', help='Kernel type for SVM')
    parser.add_argument('--probability', type=bool, default=True, help='Enable probability estimates')
    #parser.add_argument('--degree', type=int, default=3, help='Degree of the polynomial kernel function')

    # SageMaker specific arguments
    #parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    return parser.parse_args()


def load_data(args):
    """
    Loads data from the SageMaker training directory.

    Returns:
    - train_x: DataFrame with feature columns.
    - train_y: Series with target labels.
    """
    data_frames = []
    for path, _, files in os.walk(args.train):
        for file in filter(lambda x: x.lower().endswith(".csv"), files):
            file_path = os.path.join(path, file)
            data_frames.append(pd.read_csv(file_path))

    if not data_frames:
        raise ValueError(f"No files found in training directory {args.train}")
    
    train_data = pd.concat(data_frames)

    # Split into features and targets
    train_x = train_data.iloc[:, :-4]  # First 13 columns as indicators
    train_y = train_data.iloc[:, -4:]  # Last 4 columns as interventions

    return train_x, train_y


def binary_to_integer(df):
    """
    Encodes multiple binary columns into a single integer for each row.

    Returns:
    - list of integers.
    """
    binary_str = df.astype(str).agg(''.join, axis=1)
    return binary_str.apply(lambda x: int(x, 2))


def train_model(args, train_x, train_y):
    """
    Trains an SVM model on the provided data and saves it to the model directory.

    Returns:
    - best_model: Trained SVM model with the best hyperparameters.
    """
    # Parameter grid for fine-tuning
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'degree': [2, 3, 4],  # Relevant only for 'poly' kernel
        'gamma': ['scale', 'auto']  # Relevant for 'rbf' and 'poly' kernels
    }

    # Initialize GridSearchCV with SVM
    grid_search = GridSearchCV(
        SVC(probability=args.probability, random_state=42),
        param_grid,
        cv=5,
        scoring='f1_weighted',
        verbose=2
    )

    # Run grid search
    print("Starting grid search for hyperparameter tuning...")
    grid_search.fit(train_x, train_y)

    # Get the best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print("Best parameters found:", best_params)
    return best_model


def main():
    # Parse arguments and load data
    args = parse_args()
    print(args.model_dir)
    train_x, train_y_raw = load_data(args)

    # Convert multi-label binary target columns to integers for multi-class classification
    train_y = binary_to_integer(train_y_raw)

    # Train and fine-tune the SVM model
    best_model = train_model(args, train_x, train_y)

    # Save the model
    model_path = os.path.join(args.model_dir, 'svm_model.joblib')
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
