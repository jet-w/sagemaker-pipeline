
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

def get_process_args(input_data, role, processing_instance_count, pipeline_session):
    sklearn_framework_version = "1.2-1"
    sklearn_processor = SKLearnProcessor(
        framework_version=sklearn_framework_version,
        instance_type="ml.m5.large",
        instance_count=processing_instance_count,
        base_job_name="sklearn-housing-data-process",
        role=role,
        sagemaker_session=pipeline_session,
    )
    
    #processor_args = sklearn_processor.run(
    return sklearn_processor.run(
        inputs=[
            ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),
        ],
        outputs=[
            ProcessingOutput(output_name="scaler_model", source="/opt/ml/processing/scaler_model"),
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test"),
        ],
        code="steps/preprocess.py",
    )