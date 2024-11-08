from sagemaker.workflow.steps import ModelStep
from sagemaker.model import Model
from sagemaker.workflow.pipeline import Pipeline
from etc import *
# Assuming previous steps like train_step and register_step are already defined

def deploy_registered_model(pipeline_session, register_step, sklearn_estimator):
    # 1. Retrieve the model package for deployment
    # Assuming register_step is your RegisterModel step
    model_package_arn = register_step.properties.ModelPackageArn
    
    # 2. Define the Model object for the endpoint
    model = Model(
        image_uri=sklearn_estimator.training_image_uri(),
        model_data=model_package_arn,  # Use the registered model package ARN
        role=role,
        sagemaker_session=pipeline_session,
    )

    return model.deploy(
        initial_instance_count=1,
        instance_type="ml.m5.large",
        endpoint_name="SVCModelEndpoint"  # Specify a unique endpoint name
    )
#def deploy_model_artifact():
#    pass