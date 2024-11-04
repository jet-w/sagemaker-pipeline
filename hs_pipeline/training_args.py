from sagemaker.tensorflow import TensorFlow
from sagemaker.inputs import TrainingInput

def get_training_args(bucket, prefix, training_epochs, role, pipeline_session, training_instance_type, step_process):
    # Where to store the trained model
    model_path = f"s3://{bucket}/{prefix}/model/"
    
    hyperparameters = {"epochs": training_epochs}
    tensorflow_version = "2.4.1"
    python_version = "py37"
    
    tf2_estimator = TensorFlow(
        source_dir="steps",
        entry_point="training.py",
        instance_type=training_instance_type,
        instance_count=1,
        framework_version=tensorflow_version,
        role=role,
        base_job_name="tensorflow-train-model",
        output_path=model_path,
        hyperparameters=hyperparameters,
        py_version=python_version,
        sagemaker_session=pipeline_session,
    )
    
    # NOTE how the input to the training job directly references the output of the previous step.
    #train_args = tf2_estimator.fit(
    return tf2_estimator.fit(
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "test": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        }
    )