
from sagemaker.workflow.parameters import ParameterInteger, ParameterString, ParameterFloat
from sagemaker.workflow.pipeline_context import PipelineSession, LocalPipelineSession
from .data import *
#pipeline_session = PipelineSession()
pipeline_session = LocalPipelineSession()

# raw input data
input_data               = ParameterString(name="InputData", default_value=raw_s3)
# status of newly trained model in registry
model_approval_status    = ParameterString(name="ModelApprovalStatus", default_value="Approved")
# processing step parameters
processing_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.m5.xlarge")
processing_instance_count= ParameterInteger(name="ProcessingInstanceCount", default_value=1)

# training step parameters
training_instance_type   = ParameterString(name="TrainingInstanceType", default_value="ml.m5.xlarge")
training_epochs          = ParameterString(name="TrainingEpochs", default_value="100")
