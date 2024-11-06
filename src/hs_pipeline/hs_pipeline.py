from sagemaker.workflow.steps import ProcessingStep
from sagemaker.workflow.pipeline import Pipeline

from sagemaker.workflow.steps import TrainingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.properties import PropertyFile

from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo, ConditionGreaterThan
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet

from ..steps.preprocess.process_args import get_process_args
from ..steps.training.training_args import get_sklean_training_args
from ..steps.evaluation.evaluation_args import get_evaluation_args, get_svm_evaluation_args
from ..steps.register.register_args import get_register_pipeline_model

from etc import *

def get_pipeline():
    # Create a PropertyFile
    # A PropertyFile is used to be able to reference outputs from a processing step, for instance to use in a condition step.
    # For more information, visit https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-propertyfile.html
    evaluation_report = PropertyFile(
        name="EvaluationReport", output_name="evaluation", path="evaluation.json"
    )
    
    step_process = ProcessingStep(
        name="HS-mlops-PreprocessData",
        step_args=get_process_args(
            pipeline_session
        )
    )
    
    step_train_model = TrainingStep(
        name="HS-mlops-TrainModel", 
        step_args=get_sklean_training_args(
            pipeline_session, 
            step_process
        )
    )
    
    step_evaluate_model = ProcessingStep(
        name="HS-mlops-EvaluateModelPerformance",
        step_args=get_svm_evaluation_args(pipeline_session, step_process, step_train_model, 
                                          #s3_test_uri="s3://shared-hs-mlops-bucket/humansystem/preprocess/output/test", 
                                          #s3_model_uri="s3://shared-hs-mlops-bucket/humansystem/preprocess/output/pipelines-9nic0w5ptfsj-HS-mlops-TrainModel-9ZvMxL6UH9/output/"
                                        ),
        property_files=[evaluation_report],
    )
    
    # Create accuracy condition to ensure the model meets performance requirements.
    # Models with a test accuracy lower than the condition will not be registered with the model registry.
    cond_lte = ConditionLessThanOrEqualTo(
        left=JsonGet(
            step_name=step_evaluate_model.name,
            property_file=evaluation_report,
            json_path="regression_metrics.mse.value",
        ),
        right=accuracy_mse_threshold,
    )

    step_register_pipeline_model = get_register_pipeline_model(step_evaluate_model, step_evaluate_model, step_train_model)
    
    # Create a Sagemaker Pipelines ConditionStep, using the condition above.
    # Enter the steps to perform if the condition returns True / False.
    step_cond = ConditionStep(
        name="HS-mlops-MSE-Lower-Than-Threshold-Condition",
        conditions=[cond_lte],
        if_steps=[step_register_pipeline_model],  # step_register_model, step_register_scaler,
        else_steps=[],
    )
    
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
        #steps=[step_process, step_train_model, step_evaluate_model],
        #steps=[step_process, step_train_model],
        steps=[step_evaluate_model, step_cond]
    )
