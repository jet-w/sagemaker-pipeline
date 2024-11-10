from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.model_step import ModelStep

from sagemaker.model import Model
from sagemaker.sklearn.model import SKLearnModel
from sagemaker import PipelineModel

import sagemaker

from etc import *

def get_register_args(
        pipeline_session,
        step_evaluate_model, 
        step_train_model
    ):
    #svm_model_s3 = "{}/model.tar.gz".format(
    #    #"s3://shared-hs-mlops-bucket/humansystem/preprocess/output/pipelines-9nic0w5ptfsj-HS-mlops-TrainModel-9ZvMxL6UH9/output"
    #)
    #evaluation_s3_uri = "{}/evaluation.json".format(
    #    #"s3://sagemaker-us-east-1-654654179472/sagemaker-scikit-learn-2024-11-06-23-13-40-345/output/evaluation"
    #)
    
    svm_model = SKLearnModel(
        model_data=step_train_model.properties.ModelArtifacts.S3ModelArtifacts,
        role=role,
        sagemaker_session=pipeline_session,
        entry_point="steps/register/svm_process.py",
        framework_version=sklearn_framework_version,
    )
    
    pipeline_model = PipelineModel(
        models=[svm_model], role=role, sagemaker_session=pipeline_session
    )
    
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=step_evaluate_model.properties.ProcessingOutputConfig.Outputs['evaluation'].S3Output.S3Uri,
            content_type="application/json",
        )
    )

    return pipeline_model.register(
        content_types=["text/csv"],
        response_types=["application/json"],
        inference_instances=[register_instance_type],
        transform_instances=[transform_instance_type],
        model_package_group_name=model_package_group_name,
        model_metrics=model_metrics,
        approval_status=model_approval_status,
    )