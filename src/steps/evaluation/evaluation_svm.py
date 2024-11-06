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

def binary_to_integer(df):
    #encode_binary to integer
    for i in df.columns:
        ret = df[i].astype(str) if ret is None else ret + df[i].astype(str)
    return list(map(lambda x: int(x, 2), ret))

def integer_to_binary(num: int):
    return bin(num)[2:]

def get_data(data_dir):
    df_array = []
    for p, _, files in os.walk(data_dir):
        df_array.extend([pd.read_csv(os.path.join(p, file)) for file in files])
    if len(df_array) == 0:
        raise ValueError(('There are no files in {}.\n' +
                          'This usually indicates that the channel ({}) was incorrectly specified,\n' +
                          'the data specification in S3 was incorrectly specified or the role specified\n' +
                          'does not have permission to access the data.').format(args.train, "train"))
    data = pd.concat(df_array)

    x = data.iloc[:, :11]                    # Indicators
    y = binary_to_integer(data.iloc[:, 11:]) # Interventions
    return x, y

def get_model(model_dir):
    models = []
    for p, _, files in os.walk(model_dir):
        print(files)
        models.extend(list(map(lambda model: os.path.join(p, model), filter(lambda x: x=="model.tar.gz", files))))

    print(models)
    with tarfile.open(models[0], "r:gz") as tar:
        tar.extractall("./model")
    
    model_files = []
    for p, _, files in os.walk("./model"):
        print(files)
        svm_modles = filter(lambda x: x.lower().endswith(".joblib"), files)
        model_files.extend([os.path.join(p, svm) for svm in svm_modles])
    
    model_path = model_files[0]
    
    return joblib.load(model_path)    

if __name__ == "__main__":
    model_dir     = f"/opt/ml/processing/model/"
    test_data_dir = f"/opt/ml/processing/test/"
    output_dir    = f"/opt/ml/processing/evaluation"

    model = get_model(model_dir)
    x, y = get_data(test_data_dir)
    
    
    y_pred = model.predict(x)
    accuracy = accuracy_score(y, y_pred)
    
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
