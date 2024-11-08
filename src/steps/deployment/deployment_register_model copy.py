import boto3
import json
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import CreateModelStep, CreateEndpointConfigStep, CreateEndpointStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.model import Model
from sagemaker.pipeline import PipelineModel
from sagemaker.workflow.parameters import ParameterInteger, ParameterString
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet

def create_deployment_pipeline(
    role,
    model_package_group_name,
    pipeline_name,
    endpoint_name,
    instance_type="ml.m5.large",
    instance_count=1
):
    """
    Creates a SageMaker pipeline that includes model deployment steps
    """
    # Initialize pipeline session
    pipeline_session = sagemaker.workflow.pipeline_context.PipelineSession()
    region = pipeline_session.boto_region_name
    
    # Define pipeline parameters
    model_package_group_name_param = ParameterString(
        name="ModelPackageGroupName",
        default_value=model_package_group_name
    )
    
    instance_type_param = ParameterString(
        name="InstanceType",
        default_value=instance_type
    )
    
    instance_count_param = ParameterInteger(
        name="InstanceCount",
        default_value=instance_count
    )
    
    endpoint_name_param = ParameterString(
        name="EndpointName",
        default_value=endpoint_name
    )
    
    # Step to get the latest approved model package
    code = """
import boto3

def handler(context):
    client = boto3.client('sagemaker')
    response = client.list_model_packages(
        ModelPackageGroupName=context['model_package_group_name'],
        ModelApprovalStatus='Approved',
        SortBy='CreationTime',
        SortOrder='Descending',
        MaxResults=1
    )
    
    if not response['ModelPackageSummaryList']:
        raise Exception(f"No approved model packages found in group {context['model_package_group_name']}")
    
    model_package_arn = response['ModelPackageSummaryList'][0]['ModelPackageArn']
    return {'model_package_arn': model_package_arn}
"""

    get_model_step = sagemaker.workflow.steps.ProcessingStep(
        name="GetLatestApprovedModel",
        processor=sagemaker.processing.ScriptProcessor(
            image_uri=f"137112412989.dkr.ecr.{region}.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
            instance_type="ml.t3.medium",
            instance_count=1,
            base_job_name="get-model",
            role=role
        ),
        inputs=[],
        outputs=[
            sagemaker.processing.ProcessingOutput(
                output_name="model_info",
                source="/opt/ml/processing/model_info"
            )
        ],
        job_arguments=["--model-package-group-name", model_package_group_name_param],
        code=code
    )
    
    # Create model step
    model = Model(
        image_uri=pipeline_session.sagemaker_client.describe_model_package(
            ModelPackageName=get_model_step.properties.Outputs["model_info"].S3Uri
        )["InferenceSpecification"]["Containers"][0]["Image"],
        model_data=get_model_step.properties.Outputs["model_info"].S3Uri,
        role=role,
        sagemaker_session=pipeline_session
    )
    
    create_model_step = CreateModelStep(
        name="CreateModel",
        model=model
    )
    
    # Create endpoint config step
    endpoint_config_name = f"{endpoint_name_param}-config"
    create_endpoint_config_step = CreateEndpointConfigStep(
        name="CreateEndpointConfig",
        endpoint_config_name=endpoint_config_name,
        model_name=create_model_step.properties.ModelName,
        instance_type=instance_type_param,
        instance_count=instance_count_param
    )
    
    # Create/update endpoint step
    create_endpoint_step = CreateEndpointStep(
        name="CreateEndpoint",
        endpoint_name=endpoint_name_param,
        endpoint_config_name=create_endpoint_config_step.properties.EndpointConfigName
    )
    
    # Create pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            model_package_group_name_param,
            instance_type_param,
            instance_count_param,
            endpoint_name_param
        ],
        steps=[
            get_model_step,
            create_model_step,
            create_endpoint_config_step,
            create_endpoint_step
        ],
        sagemaker_session=pipeline_session
    )
    
    return pipeline

# Example of a more complex pipeline with conditional deployment
def create_conditional_deployment_pipeline(
    role,
    model_package_group_name,
    pipeline_name,
    endpoint_name,
    instance_type="ml.m5.large",
    instance_count=1,
    metric_threshold=0.85
):
    """
    Creates a pipeline that deploys model only if it meets performance threshold
    """
    pipeline_session = sagemaker.workflow.pipeline_context.PipelineSession()
    
    # Define evaluation step (assuming you have model evaluation code)
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )
    
    # Get the performance metric from evaluation
    metric_value = JsonGet(
        step_name="EvaluateModel",
        property_file=evaluation_report,
        json_path="metrics.accuracy"
    )
    
    # Create condition for deployment
    deployment_condition = ConditionGreaterThanOrEqualTo(
        left=metric_value,
        right=metric_threshold
    )
    
    # Create deployment steps
    deployment_steps = [
        create_model_step,
        create_endpoint_config_step,
        create_endpoint_step
    ]
    
    # Conditional step for deployment
    conditional_deploy_step = ConditionStep(
        name="DeployIfMetricMet",
        conditions=[deployment_condition],
        if_steps=deployment_steps,
        else_steps=[]
    )
    
    # Create pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            model_package_group_name_param,
            instance_type_param,
            instance_count_param,
            endpoint_name_param
        ],
        steps=[
            get_model_step,
            evaluation_step,
            conditional_deploy_step
        ],
        sagemaker_session=pipeline_session
    )
    
    return pipeline

# Example usage
def main():
    role = "arn:aws:iam::XXXXXXXXXXXX:role/service-role/AmazonSageMaker-ExecutionRole"
    model_package_group_name = "SklearnXgboostPipelineGroup"
    pipeline_name = "ModelDeploymentPipeline"
    endpoint_name = "pipeline-model-endpoint"
    
    # Create and register pipeline
    pipeline = create_deployment_pipeline(
        role=role,
        model_package_group_name=model_package_group_name,
        pipeline_name=pipeline_name,
        endpoint_name=endpoint_name
    )
    
    # Upsert and start pipeline
    pipeline.upsert(role_arn=role)
    execution = pipeline.start()
    
    print(f"Pipeline execution started with ARN: {execution.arn}")
    
    # Optional: Wait for completion
    execution.wait()
    print("Pipeline execution completed")

if __name__ == "__main__":
    main()