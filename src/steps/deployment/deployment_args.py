from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import ProcessingStep

from etc import *

# Define parameters for deployment
model_s3_uri = "s3://path-to-your-model-artifact/model.tar.gz"
endpoint_name = "my-endpoint-name"
instance_type = "ml.m5.large"
#role_arn = "<your-sagemaker-role-arn>"


def get_step_deployment(session, step_register):
    # Initialize the ScriptProcessor
    script_processor = ScriptProcessor(
        command=["python3"],
        instance_type=deployment_exec_instance_type,
        instance_count=1,
        role=role_arn,
        sagemaker_session=session
    )
    
    # Run the ScriptProcessor to deploy the model
    script_processor.run(
        code="./steps/deployment/deploy.py",
        arguments=[
            #"--model-s3-uri", model_s3_uri,
            #"--model-package-arn", step_register.properties.ModelPackageArn,
            "--endpoint-name", endpoint_name,
            "--instance-type", deployment_instance_type,
            "--role-arn", role_arn
        ]
    )
    
    
    #deployment_step = ProcessingStep(
    return ProcessingStep(
        name="DeployModelStep",
        processor=script_processor,
        #inputs=[],
        #outputs=[],
        job_arguments=[
            #"--model-package-arn", step_register.properties.ModelPackageArn,
            "--endpoint-name", endpoint_name,
            "--instance-type", instance_type,
            "--role-arn", role_arn
        ],
        #code="./steps/deployment/deploy.py"
    )
