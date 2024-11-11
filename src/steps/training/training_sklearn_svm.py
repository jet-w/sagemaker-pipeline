
#from __future__ import print_function

import argparse
import joblib
import os
import pandas as pd
from sklearn.svm import SVC
#import mlflow
#from sklearn.model_selection import train_test_split

def parse_args():
    parser = argparse.ArgumentParser()

    # Hyperparameters for the SVM model
    parser.add_argument('--C', type=float, default=1.0, help='Regularisation parameter')
    parser.add_argument('--kernel', type=str, default='linear', help='Kernel type for SVM')
    parser.add_argument('--probability', type=bool, default=True, help='Enable probability estimates')
    parser.add_argument('--epochs', type=int, default=0, help='Enable probability estimates')

    # SageMaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    #parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])

    return parser.parse_args()

def binary_to_integer(df):
    #encode_binary to integer
    ret = None
    for i in df.columns:
        ret = df[i].astype(str) if ret is None else ret + df[i].astype(str)
    
    return list(map(lambda x: int(x, 2), ret))

def get_data(args):
    df_array = []
    for p, _, files in os.walk(args.train):
        df_array.extend([pd.read_csv(os.path.join(p, file)) for file in files])

    if len(df_array) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    elif len(df_array) == 1:
        train_data = df_array[0]
    else:
        train_data = pd.concat(df_array)

    train_x = train_data.iloc[:, :11] # Indicators
    train_y = binary_to_integer(train_data.iloc[:, 11:]) # Interventions
    
    return train_x, train_y

def train(args, train_x, train_y):
    # Retrieve hyperparameters from the arguments
    C = args.C
    kernel = args.kernel
    probability = args.probability

    # Output Path
    model_path = args.model_dir

    # Create and train the SVM model
    #with mlflow.start_run():
    clf = SVC(C=C, kernel=kernel, probability=probability, random_state=42)
    clf.fit(train_x, train_y)
    joblib.dump(clf, os.path.join(model_path, 'svm_model.joblib'))

if __name__ == '__main__':
    args = parse_args()
    train_x, train_y = get_data(args)
    train(args, train_x, train_y)