
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

from sagemaker.workflow.steps import ProcessingStep
from sagemaker.processing import Processor
from etc import input_data, role, processing_instance_count, preprocessing_instance_type, bucket
import logging
from etc import *


endpoint_name = "HS-endpoint-Intervention-Recommendation"
def get_step_deployment(pipeline_session, step_register, pkg_arn = None):
    s3_train  = f"s3://{bucket}/humansystem/preprocess/output/train"
    s3_test   = f"s3://{bucket}/humansystem/preprocess/output/test"

    sklearn_framework_version = "1.2-1"
    sklearn_processor = SKLearnProcessor(
        framework_version=sklearn_framework_version,
        instance_type=preprocessing_instance_type,
        instance_count=processing_instance_count,
        base_job_name="sklearn-housing-data-process",
        role=role,
        sagemaker_session=pipeline_session,
        #image_uri=""
    )
    pkg_arn = pkg_arn if pkg_arn else step_register.properties.ModelPackageArn
    #processor_args = sklearn_processor.run(
    args = sklearn_processor.run(
        inputs=[
            ProcessingInput(source=input_data, destination="/opt/ml/processing/input"),
        ],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/train", destination=s3_train),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/test", destination=s3_test),
        ],
        code="./steps/deployment/deploy.py",
        
        arguments=[
            "--model-package-arn", pkg_arn,
            "--endpoint-name", endpoint_name,
            "--instance-type", deployment_instance_type,
            "--role-arn", role_arn
        ]
    )


    return ProcessingStep(
        name="HS-mlops-Deployment",
        step_args=args
    )

#from sagemaker.processing import ScriptProcessor
#from sagemaker.workflow.parameters import ParameterString
#from sagemaker.workflow.steps import ProcessingStep
#
#from etc import *
#
## Define parameters for deployment
##model_s3_uri = "s3://path-to-your-model-artifact/model.tar.gz"
##endpoint_name = "my-endpoint-name"
#instance_type = "ml.m5.large"
##role_arn = "<your-sagemaker-role-arn>"
#endpoint_name = "HS-endpoint-Intervention-Recommendation"
#
#def get_step_deployment(session, step_register):
#    # Initialize the ScriptProcessor
#    script_processor = ScriptProcessor(
#        
#        command=["python3"],
#        instance_type=deployment_exec_instance_type,
#        instance_count=1,
#        role=role,
#        sagemaker_session=session
#    )
#    
#    # Run the ScriptProcessor to deploy the model
#    script_processor.run(
#        code="./steps/deployment/deploy.py",
#        #arguments=[
#        #    #"--model-s3-uri", model_s3_uri,
#        #    #"--model-package-arn", step_register.properties.ModelPackageArn,
#        #    "--endpoint-name", endpoint_name,
#        #    "--instance-type", deployment_instance_type,
#        #    "--role-arn", role_arn
#        #]
#    )
#    
#    
#    #deployment_step = ProcessingStep(
#    return ProcessingStep(
#        name="DeployModelStep",
#        processor=script_processor,
#        #inputs=[],
#        #outputs=[],
#        #job_arguments=[
#        #    #"--model-package-arn", step_register.properties.ModelPackageArn,
#        #    "--endpoint-name", endpoint_name,
#        #    "--instance-type", instance_type,
#        #    "--role-arn", role_arn
#        #],
#        #code="./steps/deployment/deploy.py"
#    )
#    return None

