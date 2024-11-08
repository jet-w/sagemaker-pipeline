import os
import pandas as pd
import StringIO
import tarfile
import joblib
from etc import label_column

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
    models = []
    for p, _, files in os.walk(model_dir):
        models.extend(list(map(lambda model: os.path.join(p, model), filter(lambda x: x=="model.tar.gz", files))))

    with tarfile.open(models[0], "r:gz") as tar:
        tar.extractall("./model")
    
    model_files = []
    for p, _, files in os.walk("./model"):
        svm_modles = filter(lambda x: x.lower().endswith(".joblib"), files)
        model_files.extend([os.path.join(p, svm) for svm in svm_modles])
    
    model_path = model_files[0]
    
    return joblib.load(model_path)

def input_fn(request_body, request_content_type):
    if request_content_type == "text/csv":
        df = pd.read_csv(StringIO(request_body))
        return df
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    ret = []
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