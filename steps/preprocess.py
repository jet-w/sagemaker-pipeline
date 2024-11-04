import pandas as pd
import sys
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tarfile

try:
    from sagemaker_containers.beta.framework import (
        content_types, encoders, env, modules,
        transformer,   worker,   server,
    )
except ImportError:
    pass
from etc import *

if __name__ == "__main__":
    df = pd.read_csv(f"{base_dir}/input/raw_data_all.csv")
    feature_data = df.drop(label_column, axis=1, inplace=False)
    label_data = df[label_column]
    x_train, x_test, y_train, y_test = train_test_split(feature_data, label_data, test_size=0.33)

    scaler = StandardScaler()

    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    train_dataset = pd.concat([pd.DataFrame(x_train), y_train.reset_index(drop=True)], axis=1)
    test_dataset = pd.concat([pd.DataFrame(x_test), y_test.reset_index(drop=True)], axis=1)

    train_dataset.columns = feature_columns + [label_column]
    test_dataset.columns = feature_columns + [label_column]

    train_dataset.to_csv(f"{base_dir}/train/train.csv", header=True, index=False)
    test_dataset.to_csv(f"{base_dir}/test/test.csv", header=True, index=False)
    joblib.dump(scaler, "model.joblib")
    with tarfile.open(f"{base_dir}/scaler_model/model.tar.gz", "w:gz") as tar_handle:
        tar_handle.add(f"model.joblib")