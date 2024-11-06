import json
import numpy as np
import pandas as pd
import pathlib
import tarfile
import joblib

import os
import sys
from sklearn.metrics import accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

label_column    = ["Peer-Work", "Reflection", "Additional-Resources", "Reminders"]

def get_data(args):
    df_array = []
    for p, _, files in os.walk("data"):
        df_array.extend([pd.read_csv(os.path.join(p, file)) for file in files])
    if len(df_array) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    train_data = pd.concat(df_array)

    train_x = train_data.iloc[:, :11] # Indicators
    train_y = train_data.iloc[:, 11:] # Interventions
    ret = None

    #encode_binary to integer
    for i in train_y.columns:
        ret = train_y[i].astype(str) if ret is None else ret + train_y[i].astype(str)
    
    train_y = list(map(lambda x: int(x, 2), ret))
    return train_x, train_y

if __name__ == "__main__":
    model_dir     = f"/opt/ml/processing/model/"
    test_data_dir = f"/opt/ml/processing/test/"
    output_dir    = f"/opt/ml/processing/evaluation"

    model_files = []
    for p, _, files in os.walk(model_dir):
        svm_modles = filter(lambda x: x == "svm_model.joblib", files)
        model_files.extend([os.path.join(p, svm) for svm in svm_modles])
    
    model_path = model_files[0]
    
    model = joblib.load(model_path)

    

    accuracy = accuracy_score(y_test, y_pred)
    
    # Available metrics to add to model: https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-model-quality-metrics.html
    report_dict = {
        "regression_metrics": {
            "mse": {"value": accuracy, "standard_deviation": "NaN"},
            "accuracy": {"value": accuracy, "standard_deviation": "NaN"},
        },
    }

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))