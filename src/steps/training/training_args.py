from sagemaker.tensorflow import TensorFlow
from sagemaker.pytorch import PyTorch
from sagemaker.sklearn import SKLearn

from sagemaker.inputs import TrainingInput
import os
from etc import *

def get_tensorflow_training_args(pipeline_session, step_process):
    tf2_estimator = TensorFlow(
        source_dir       = "steps/training",
        entry_point      = "training_sklearn_sample.py",
        instance_type    = training_instance_type,
        instance_count   = 1,
        framework_version= tensorflow_version,
        role             = role,
        base_job_name    = "tensorflow-train-model",
        output_path      = model_path,
        hyperparameters  = hyperparameters,
        py_version       = python_version,
        sagemaker_session= pipeline_session,
    )

    return tf2_estimator.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "test": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                content_type="text/csv",
            )
        }
    )

def get_sklean_training_args(pipeline_session, step_process):
    sklearn_estimator = SKLearn(
        source_dir       = "steps/training",
        entry_point      = "training_sklearn_svm.py",
        framework_version= "1.2-1",
        instance_type    = training_instance_type,
        role             = role,
        sagemaker_session=pipeline_session,
        hyperparameters  = hyperparameters,
        py_version       = "py3",
        keep_alive_period_in_seconds=3600,
        output_path      = "s3://test/output"#s3_model_output
    )

    sk_model = sklearn_estimator.create_model()

    print(sk_model.model_data)
    print(sk_model.role)
    print(sk_model.sagemaker_session)
    print(sk_model.entry_point)
    print(sk_model.framework_version)
    
    return sklearn_estimator.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "test": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                content_type="text/csv",
            )
        }
    )

def get_pytorch_rnn_training_args(pipeline_session, step_process):
    # Configure the training job
    hyperparameters = {
        'epochs': 10,
        'batch-size': 32,
        'learning-rate': 0.001,
        'hidden-dim': 64,
        'num-layers': 2
    }
    
    estimator = PyTorch(
        source_dir       = "steps",
        entry_point      = 'train_pytorch_rnn.py',
        role             = role,
        instance_count   = 1,
        instance_type    = training_instance_type, #'ml.p3.2xlarge',  # GPU instance
        framework_version='2.0.1',
        py_version       ='py39',
        output_path      = model_path,
        hyperparameters  =hyperparameters,
        sagemaker_session= pipeline_session,
    )
    model_s3 = os.path.dirname(step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri)
    model_s3 = f"{model_s3}/model"
    # Start training
    return estimator.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "test": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "model": TrainingInput(
                s3_data=model_s3,
                content_type="text/csv",
            ),
        }
    )

