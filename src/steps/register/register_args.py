from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.model_step import ModelStep

from sagemaker.model import Model
from sagemaker.sklearn.model import SKLearnModel
from sagemaker import PipelineModel

import sagemaker

from etc import *

def get_register_args(
        step_evaluate_model, 
        step_train_model
    ):
    print("*"*100)
    print(step_train_model.properties.ModelArtifacts.S3ModelArtifacts)
    print("#"*100)
    svm_model_s3 = "{}/model.tar.gz".format(
        #step_process.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
        step_train_model.properties.ModelArtifacts.S3ModelArtifacts
        #"s3://shared-hs-mlops-bucket/humansystem/preprocess/output/pipelines-9nic0w5ptfsj-HS-mlops-TrainModel-9ZvMxL6UH9/output"
    )
    
    svm_model = SKLearnModel(
        model_data=svm_model_s3,
        role=role,
        sagemaker_session=pipeline_session,
        entry_point="steps/register/svm_process.py",
        framework_version=sklearn_framework_version,
    )
    
    pipeline_model = PipelineModel(
        models=[svm_model], role=role, sagemaker_session=pipeline_session
    )

    evaluation_s3_uri = "{}/evaluation.json".format(
        step_evaluate_model.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
        #"s3://sagemaker-us-east-1-654654179472/sagemaker-scikit-learn-2024-11-06-23-13-40-345/output/evaluation"
    )
    
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=evaluation_s3_uri,
            content_type="application/json",
        )
    )
    
    #register_args = pipeline_model.register(
    return pipeline_model.register(
        content_types=["text/csv"],
        response_types=["application/json"],
        inference_instances=["ml.m5.large"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        model_metrics=model_metrics,
        approval_status=model_approval_status,
    )