from .pipeline import *

tensorflow_version = "2.4.1"
sklearn_framework_version = "1.2-1"

hyperparameters = {"epochs": training_epochs}
tensorflow_version = "2.4.1"
python_version = "py37"


# Where to store the trained model
model_path = f"s3://{bucket}/humansystem/models/predict/"