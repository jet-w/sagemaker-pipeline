from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.model_step import ModelStep

from sagemaker.model import Model
from sagemaker.sklearn.model import SKLearnModel
from sagemaker import PipelineModel

import sagemaker

from etc import *

def get_register_pipeline_model(
        step_process, 
        step_evaluate_model, 
        step_train_model
    ):
    
    svm_model_s3 = "{}/model.tar.gz".format(
        #step_process.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
        "s3://shared-hs-mlops-bucket/humansystem/preprocess/output/pipelines-9nic0w5ptfsj-HS-mlops-TrainModel-9ZvMxL6UH9/output/"
    )
    
    svm_model = SKLearnModel(
        model_data=svm_model_s3,
        role=role,
        sagemaker_session=pipeline_session,
        entry_point="steps/preprocess/preprocess.py",
        framework_version=sklearn_framework_version,
    )
    
    #tf_model_image_uri = sagemaker.image_uris.retrieve(
    #    framework="tensorflow",
    #    region=region,
    #    version=tensorflow_version,
    #    image_scope="inference",
    #    py_version="py37",
    #    instance_type="ml.m5.xlarge",
    #)
    #
    #tf_model = Model(
    #    image_uri=tf_model_image_uri,
    #    model_data=step_train_model.properties.ModelArtifacts.S3ModelArtifacts,
    #    sagemaker_session=pipeline_session,
    #    role=role,
    #)
    
    pipeline_model = PipelineModel(
        models=[svm_model, tf_model], role=role, sagemaker_session=pipeline_session
    )

    evaluation_s3_uri = "{}/evaluation.json".format(
        step_evaluate_model.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
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
        response_types=["text/csv"],
        inference_instances=["ml.m5.large", "ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        model_metrics=model_metrics,
        approval_status=model_approval_status,
    )
    
    #step_register_pipeline_model = ModelStep(
    #return ModelStep(
    #    name="PipelineModel",
    #    step_args=register_args,
    #)