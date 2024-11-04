from __future__ import print_function

import argparse
import joblib
import os
import pandas as pd
from sklearn.svm import SVC
import mlflow
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters for the SVM model
    parser.add_argument('--C', type=float, default=1.0, help='Regularisation parameter')
    parser.add_argument('--kernel', type=str, default='linear', help='Kernel type for SVM')
    parser.add_argument('--probability', type=bool, default=True, help='Enable probability estimates')

    # SageMaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    # Take the set of files and read them all into a single pandas DataFrame
    input_files = [os.path.join(args.train, file) for file in os.listdir(args.train) if os.path.isfile(os.path.join(args.train, file))]
    if len(input_files) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    raw_data = [pd.read_csv(file, header=None, engine="python") for file in input_files]
    train_data = pd.concat(raw_data)

    # Set the Tracking Server URI using the ARN of the Tracking Server you created
    mlflow.set_tracking_uri(os.environ.get('MLFLOW_TRACKING_ARN', ''))
    
    # Enable autologging in MLflow
    mlflow.autolog()

   # Separating Indicators and Interventions
    features = data.iloc[:, :11] # Indicators
    targets = data.iloc[:, 11:] # Interventions

    # Retrieve hyperparameters from the arguments
    C = args.C
    kernel = args.kernel
    probability = args.probability

    # Create and train the SVM model
    with mlflow.start_run():
        clf = SVC(C=C, kernel=kernel, probability=probability, random_state=42)
        clf.fit(train_X, train_y)

        # Save the model to the specified directory
        joblib.dump(clf, os.path.join(args.model_dir, 'model.joblib'))
        print(f"Model saved to {os.path.join(args.model_dir, 'model.joblib')}")
