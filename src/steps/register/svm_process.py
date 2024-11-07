import os
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import StringIO
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
    prediction = model.predict(input_data)
    [integer_to_binary(i) for i in prediction]
    transformed_data = model.transform(input_data)
    return transformed_data

def output_fn(prediction, accept):
    if accept == "text/csv":
        return ','.join(map(str, prediction.flatten()))
    raise ValueError(f"Unsupported accept type: {accept}")