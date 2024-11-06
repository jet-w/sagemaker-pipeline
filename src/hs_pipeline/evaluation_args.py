import sagemaker
from sagemaker.sklearn.processing import ScriptProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn import SKLearnProcessor
from sagemaker.workflow.steps import ProcessingStep
from etc import *

def get_evaluation_args(pipeline_session, step_process, step_train_model):
    tf_eval_image_uri = sagemaker.image_uris.retrieve(
        framework="tensorflow",
        version=tensorflow_version,
        instance_type="ml.m5.xlarge",
        region=region,
        image_scope="training",
        py_version="py37",
    )
    
    evaluate_model_processor = ScriptProcessor(
        role=role,
        image_uri=tf_eval_image_uri,
        command=["python3"],
        instance_count=1,
        instance_type=processing_instance_type,
        sagemaker_session=pipeline_session,
    )
    return evaluate_model_processor.run(
        inputs=[
            ProcessingInput(
                source=step_train_model.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
        ],
        code="steps/evaluation/evaluation_svm.py",
    )



def get_svm_evaluation_args(pipeline_session, step_process, step_train_model):
    #tf_eval_image_uri = sagemaker.image_uris.retrieve(
    #    framework="sklearn",
    #    version=tensorflow_version,
    #    instance_type="ml.m5.xlarge",
    #    region=region,
    #    image_scope="training",
    #    py_version="py37",
    #)
    
    #evaluate_model_processor = ScriptProcessor(
    #    role=role,
    #    image_uri=tf_eval_image_uri,
    #    command=["python3"],
    #    instance_count=1,
    #    instance_type=processing_instance_type,
    #    sagemaker_session=pipeline_session,
    #)
    s3_test_uri = step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri
    s3_model_uri = step_train_model.properties.ModelArtifacts.S3ModelArtifacts

    # Define the SKLearnProcessor for evaluation
    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type="ml.m5.large",
        instance_count=1,
        role=role
    )
    
    # Define the Processing Step for evaluation
    #evaluation_step = ProcessingStep(
    #    name="ModelEvaluation",
    #    processor=sklearn_processor,
    #    inputs=[
    #        ProcessingInput(source=train_job.model_artifacts, destination="/opt/ml/processing/model"),
    #        ProcessingInput(source=test_data_s3_uri, destination="/opt/ml/processing/test")
    #    ],
    #    outputs=[
    #        ProcessingOutput(destination="s3://your-s3-bucket/evaluation", output_name="evaluation", source="/opt/ml/output/evaluation")
    #    ],
    #    code="evaluate.py"
    #)


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