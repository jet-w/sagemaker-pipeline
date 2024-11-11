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
from sagemaker import ModelPackage

from steps.preprocess.process_args import get_step_preprocess
from steps.training.training_args import get_step_training
from steps.evaluation.evaluation_args import get_step_evaluation
from steps.register.register_args import get_step_register
from steps.deployment.deployment_args import get_step_deployment

from sagemaker.workflow.functions import Join
#from steps.deployment.deployment_args import get_deployment_args
from etc import *


def get_step_conditional(step_name, evaluation_report, register_step, deployment_step):
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
        if_steps=[register_step, deployment_step],  # step_register_model, step_register_scaler,
        else_steps=[register_step],
    )


def get_pipeline():
    # Create a PropertyFile
    # A PropertyFile is used to be able to reference outputs from a processing step, for instance to use in a condition step.
    # For more information, visit https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-propertyfile.html
    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )
    
    step_process = get_step_preprocess(pipeline_session)
    step_train_model, estimator = get_step_training(pipeline_session, step_process)
    step_evaluate_model = get_step_evaluation(pipeline_session, step_process, step_train_model, evaluation_report)
    step_register =get_step_register(pipeline_session, step_evaluate_model, step_train_model)

    pkg_arn = "arn:aws:sagemaker:us-east-1:654654179472:model-package/HumanSystemsAIOpsModelPackageGroup/8"
    step_deployment = get_step_deployment(pipeline_session, step_register, pkg_arn)
    step_conditional = get_step_conditional(step_evaluate_model.name, evaluation_report, step_register, step_deployment)

    # Create a Sagemaker Pipeline.
    # Each parameter for the pipeline must be set as a parameter explicitly when the pipeline is created.
    # Also pass in each of the steps created above.
    # Note that the order of execution is determined from each step's dependencies on other steps,
    # not on the order they are passed in below.
    return Pipeline(
        name=pipeline_name,
        parameters=[
            training_instance_type,
            "ml.t3.large",#processing_instance_type,
            processing_instance_count,
            input_data,
            model_approval_status,
            training_epochs,
            accuracy_mse_threshold,
        ],
        #steps=[step_process, step_train_model, step_evaluate_model, step_cond],
        steps=[step_process, step_train_model, step_evaluate_model, step_register]#, step_deployment],
        #steps=[step_process, step_train_model, step_evaluate_model, step_cond]
        #steps=[step_process, step_train_model],
        #steps=[step_evaluate_model, step_cond]
        #steps = [register_step]
        #steps = [step_deployment]
    )
