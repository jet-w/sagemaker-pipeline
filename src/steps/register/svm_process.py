import os
import pandas as pd
from io import StringIO
import tarfile
import joblib

label_column    = ["Peer-Work", "Reflection", "Additional-Resources", "Reminders"]

def binary_to_integer(df):
    #encode_binary to integer
    ret = None
    for i in df.columns:
        ret = df[i].astype(str) if ret is None else ret + df[i].astype(str)
    return list(map(lambda x: int(x, 2), ret))

def integer_to_binary(num: int):
    ret = bin(num)[2:]
    if len(ret) < len(label_column):
        ret = f"{'0'*(len(label_column) - len(ret))}{ret}"
    return ret

def model_fn(model_dir):
    tar_files = []
    model_files = []
    print("model directory:", model_dir)
    for p, _, files in os.walk(model_dir):
        print("Model Directory Files:", p, files)
        tar_files.extend(list(map(lambda model: os.path.join(p, model), filter(lambda x: x.endswith(".tar.gz"), files))))
        model_files.extend(list(map(lambda model: os.path.join(p, model), filter(lambda x: x.endswith(".joblib"), files))))
    if len(tar_files) > 0:
        with tarfile.open(tar_files[0], "r:gz") as tar:
            tar.extractall("./model")
    for p, _, files in os.walk("./model"):
        print("Decompressed Files:", files)
        model_files.extend([os.path.join(p, svm) for svm in  filter(lambda x: x.lower().endswith(".joblib"), files)])
    print("Final Model Files:", model_files)
    return joblib.load(model_files[0]) if len(model_files) > 0 else None

def input_fn(request_body, request_content_type):
    print("request_data:", request_body)
    if request_content_type == "text/csv":
        df = pd.read_csv(StringIO(request_body))
        return df
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    ret = []
    print("Predict - Input Data:", input_data)
    print("Predict - Model:", model)
    prediction = model.predict(input_data)
    for binary in [integer_to_binary(i) for i in prediction]:
        obj = {}
        for idx, val in enumerate(binary):
            obj[label_column[idx]] = val
        ret.append(obj)
    return ret

#def output_fn(prediction, accept):
#    if accept == "application/json":
#        return ','.join(map(str, prediction.flatten()))
#    raise ValueError(f"Unsupported accept type: {accept}")