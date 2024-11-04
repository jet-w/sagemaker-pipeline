import sagemaker
from sagemaker.sklearn.processing import ScriptProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
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
        code="steps/evaluation.py",
    )
