
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from etc import input_data, role, processing_instance_count, bucket
import logging

def get_process_args(pipeline_session):
    s3_train  = f"s3://{bucket}/humansystem/preprocess/output/train"
    s3_test   = f"s3://{bucket}/humansystem/preprocess/output/test"

    sklearn_framework_version = "1.2-1"
    sklearn_processor = SKLearnProcessor(
        framework_version=sklearn_framework_version,
        instance_type="ml.m5.large",
        instance_count=processing_instance_count,
        base_job_name="sklearn-housing-data-process",
        role=role,
        sagemaker_session=pipeline_session,
        #image_uri=""
    )
    
    #processor_args = sklearn_processor.run(
    return sklearn_processor.run(
        inputs=[
            ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train", destination=s3_train),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test", destination=s3_test),
        ],
        code="steps/preprocess/preprocess.py",
    )