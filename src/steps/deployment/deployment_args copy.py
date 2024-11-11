import sagemaker
from sagemaker.sklearn.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.processing import ScriptProcessor, FrameworkProcessor
from sagemaker.sklearn import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep
from etc import *

def get_deployment_args(pipeline_session, step_process, step_train_model, s3_test_uri=None, s3_model_uri=None):
    s3_test_uri = step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri if s3_test_uri is None else s3_test_uri 
    s3_model_uri = step_train_model.properties.ModelArtifacts.S3ModelArtifacts if s3_model_uri is None else s3_model_uri

    # Define the SKLearnProcessor for evaluation
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type=evaluation_instance_type,
        instance_count=1,
        role=role,
        sagemaker_session=pipeline_session,
    )

    return sklearn_processor.run(
        inputs=[
            ProcessingInput(
                source=s3_model_uri,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=s3_test_uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code="steps/evaluation/evaluation_svm.py",
    )