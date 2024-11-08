from sagemaker.workflow.pipeline import Pipeline


from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.condition_step import ConditionStep


from sagemaker.model import Model
from sagemaker.workflow.properties import PropertyFile

from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo, ConditionGreaterThan
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet

from steps.preprocess.process_args import get_process_args
from steps.training.training_args import get_sklean_training_args
from steps.evaluation.evaluation_args import get_evaluation_args, get_svm_evaluation_args
from steps.register.register_args import get_register_args
from steps.deployment.deployment_args import get_deployment_args
from etc import *

def get_step_preprocess(session):
    return ProcessingStep(
        name="HS-mlops-PreprocessData",
        step_args=get_process_args(
            session
        )
    )

def get_step_training(session, step_process):
    step_args, estimator =get_sklean_training_args(session, step_process)
    return TrainingStep(
        name="HS-mlops-TrainModel", 
        step_args=step_args
    ), estimator

def get_step_evaluation(step_process, step_train_model, evaluation_report):
    return ProcessingStep(
        name="HS-mlops-EvaluateModelPerformance",
        step_args=get_svm_evaluation_args(
            pipeline_session, step_process, step_train_model, 
                                          #s3_test_uri="s3://shared-hs-mlops-bucket/humansystem/preprocess/output/test", 
                                          #s3_model_uri="s3://shared-hs-mlops-bucket/humansystem/preprocess/output/pipelines-9nic0w5ptfsj-HS-mlops-TrainModel-9ZvMxL6UH9/output/"
                                        ),
        property_files=[evaluation_report],
    )

def get_step_register(step_evaluate_model, step_train_model):
    return ModelStep(
        name="HS-RegisterModel",
        step_args=get_register_args(step_evaluate_model, step_train_model),
    )

def get_step_deployment(sklearn_estimator, step_train_model):
    # Define a model object
    model = Model(
        image_uri=sklearn_estimator.training_image_uri(),
        model_data=step_train_model.properties.ModelArtifacts.S3ModelArtifacts,  # or register_step.properties.ModelPackageArn for a registered model
        role=role,
        sagemaker_session=sagemaker_session
    )
    
    # Define the deployment step
    return ModelStep(
        name="DeployModel",
        model=model,
        instance_type="ml.m5.large",
        initial_instance_count=1,
        endpoint_name="sagemaker-pipeline-endpoint"
    )

def get_step_conditional(step_name, evaluation_report, register_step):
    # Create accuracy condition to ensure the model meets performance requirements.
    # Models with a test accuracy lower than the condition will not be registered with the model registry.
    #cond_lte = ConditionLessThanOrEqualTo(
    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step_name=step_name, #step_evaluate_model.name,
            property_file=evaluation_report,
            json_path="regression_metrics.mse.value",
        ),
        right=accuracy_mse_threshold,
    )
    #step_cond = ConditionStep(
    return ConditionStep(
        name="HS-mlops-MSE-Lower-Than-Threshold-Condition",
        conditions=[cond_lte],
        if_steps=[register_step],  # step_register_model, step_register_scaler,
        else_steps=[],
    )


def get_pipeline():
    # Create a PropertyFile
    # A PropertyFile is used to be able to reference outputs from a processing step, for instance to use in a condition step.
    # For more information, visit https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-propertyfile.html
    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )
    
    step_process = get_step_preprocess()
    step_train_model, estimator = get_step_training(step_process)
    step_evaluate_model = get_step_evaluation(step_process, step_train_model, evaluation_report)
    register_step =get_step_register(step_evaluate_model, step_train_model)
    step_deployment = get_step_deployment(estimator, step_train_model)
    step_conditional = get_step_conditional(step_evaluate_model.name, evaluation_report, register_step)

    # Create a Sagemaker Pipeline.
    # Each parameter for the pipeline must be set as a parameter explicitly when the pipeline is created.
    # Also pass in each of the steps created above.
    # Note that the order of execution is determined from each step's dependencies on other steps,
    # not on the order they are passed in below.
    return Pipeline(
        name=pipeline_name,
        parameters=[
            training_instance_type,
            processing_instance_type,
            processing_instance_count,
            input_data,
            model_approval_status,
            training_epochs,
            accuracy_mse_threshold,
        ],
        #steps=[step_process, step_train_model, step_evaluate_model, step_cond],
        steps=[step_process, step_train_model, step_evaluate_model, step_deployment],
        #steps=[step_process, step_train_model, step_evaluate_model, step_cond]
        #steps=[step_process, step_train_model],
        #steps=[step_evaluate_model, step_cond]
        #steps = [register_step]
    )
