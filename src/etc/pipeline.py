
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
preprocessing_instance_type = "ml.t3.medium"#"ml.m5.large"#ParameterString(name="ProcessingInstanceType", default_value="ml.m5.large")
processing_instance_count= ParameterInteger(name="ProcessingInstanceCount", default_value=1)

# training step parameters
training_instance_type   = ParameterString(name="TrainingInstanceType", default_value="ml.t2.medium")
training_epochs          = ParameterString(name="TrainingEpochs", default_value="100")

evaluation_instance_type = "ml.t3.medium"#ParameterString(name="ProcessingInstanceType", default_value="ml.t2.medium")
# [ml.t3.medium, ml.t3.large, ml.t3.xlarge, ml.t3.2xlarge, ml.m4.xlarge, ml.m4.2xlarge, ml.m4.4xlarge, ml.m4.10xlarge, ml.m4.16xlarge, ml.c4.xlarge, ml.c4.2xlarge, ml.c4.4xlarge, ml.c4.8xlarge, ml.p2.xlarge, ml.p2.8xlarge, ml.p2.16xlarge, ml.p3.2xlarge, ml.p3.8xlarge, ml.p3.16xlarge, ml.c5.xlarge, ml.c5.2xlarge, ml.c5.4xlarge, ml.c5.9xlarge, ml.c5.18xlarge, ml.m5.large, ml.m5.xlarge, ml.m5.2xlarge, ml.m5.4xlarge, ml.m5.12xlarge, ml.m5.24xlarge, ml.r5.large, ml.r5.xlarge, ml.r5.2xlarge, ml.r5.4xlarge, ml.r5.8xlarge, ml.r5.12xlarge, ml.r5.16xlarge, ml.r5.24xlarge, ml.g4dn.xlarge, ml.g4dn.2xlarge, ml.g4dn.4xlarge, ml.g4dn.8xlarge, ml.g4dn.12xlarge, ml.g4dn.16xlarge, ml.g5.xlarge, ml.g5.2xlarge, ml.g5.4xlarge, ml.g5.8xlarge, ml.g5.16xlarge, ml.g5.12xlarge, ml.g5.24xlarge, ml.g5.48xlarge, ml.r5d.large, ml.r5d.xlarge, ml.r5d.2xlarge, ml.r5d.4xlarge, ml.r5d.8xlarge, ml.r5d.12xlarge, ml.r5d.16xlarge, ml.r5d.24xlarge]

register_instance_type   = "ml.t2.medium"#ParameterString(name="ProcessingInstanceType", default_value="ml.t2.medium")
# [ml.t2.medium, ml.t2.large, ml.t2.xlarge, ml.t2.2xlarge, ml.m4.xlarge, ml.m4.2xlarge, ml.m4.4xlarge, ml.m4.10xlarge, ml.m4.16xlarge, ml.m5.large, ml.m5.xlarge, ml.m5.2xlarge, ml.m5.4xlarge, ml.m5.12xlarge, ml.m5.24xlarge, ml.m5d.large, ml.m5d.xlarge, ml.m5d.2xlarge, ml.m5d.4xlarge, ml.m5d.12xlarge, ml.m5d.24xlarge, ml.c4.large, ml.c4.xlarge, ml.c4.2xlarge, ml.c4.4xlarge, ml.c4.8xlarge, ml.p2.xlarge, ml.p2.8xlarge, ml.p2.16xlarge, ml.p3.2xlarge, ml.p3.8xlarge, ml.p3.16xlarge, ml.c5.large, ml.c5.xlarge, ml.c5.2xlarge, ml.c5.4xlarge, ml.c5.9xlarge, ml.c5.18xlarge, ml.c5d.large, ml.c5d.xlarge, ml.c5d.2xlarge, ml.c5d.4xlarge, ml.c5d.9xlarge, ml.c5d.18xlarge, ml.g4dn.xlarge, ml.g4dn.2xlarge, ml.g4dn.4xlarge, ml.g4dn.8xlarge, ml.g4dn.12xlarge, ml.g4dn.16xlarge, ml.r5.large, ml.r5.xlarge, ml.r5.2xlarge, ml.r5.4xlarge, ml.r5.12xlarge, ml.r5.24xlarge, ml.r5d.large, ml.r5d.xlarge, ml.r5d.2xlarge, ml.r5d.4xlarge, ml.r5d.12xlarge, ml.r5d.24xlarge, ml.inf1.xlarge, ml.inf1.2xlarge, ml.inf1.6xlarge, ml.inf1.24xlarge, ml.dl1.24xlarge, ml.c6i.large, ml.c6i.xlarge, ml.c6i.2xlarge, ml.c6i.4xlarge, ml.c6i.8xlarge, ml.c6i.12xlarge, ml.c6i.16xlarge, ml.c6i.24xlarge, ml.c6i.32xlarge, ml.g5.xlarge, ml.g5.2xlarge, ml.g5.4xlarge, ml.g5.8xlarge, ml.g5.12xlarge, ml.g5.16xlarge, ml.g5.24xlarge, ml.g5.48xlarge, ml.g6.xlarge, ml.g6.2xlarge, ml.g6.4xlarge, ml.g6.8xlarge, ml.g6.12xlarge, ml.g6.16xlarge, ml.g6.24xlarge, ml.g6.48xlarge, ml.p4d.24xlarge, ml.c7g.large, ml.c7g.xlarge, ml.c7g.2xlarge, ml.c7g.4xlarge, ml.c7g.8xlarge, ml.c7g.12xlarge, ml.c7g.16xlarge, ml.m6g.large, ml.m6g.xlarge, ml.m6g.2xlarge, ml.m6g.4xlarge, ml.m6g.8xlarge, ml.m6g.12xlarge, ml.m6g.16xlarge, ml.m6gd.large, ml.m6gd.xlarge, ml.m6gd.2xlarge, ml.m6gd.4xlarge, ml.m6gd.8xlarge, ml.m6gd.12xlarge, ml.m6gd.16xlarge, ml.c6g.large, ml.c6g.xlarge, ml.c6g.2xlarge, ml.c6g.4xlarge, ml.c6g.8xlarge, ml.c6g.12xlarge, ml.c6g.16xlarge, ml.c6gd.large, ml.c6gd.xlarge, ml.c6gd.2xlarge, ml.c6gd.4xlarge, ml.c6gd.8xlarge, ml.c6gd.12xlarge, ml.c6gd.16xlarge, ml.c6gn.large, ml.c6gn.xlarge, ml.c6gn.2xlarge, ml.c6gn.4xlarge, ml.c6gn.8xlarge, ml.c6gn.12xlarge, ml.c6gn.16xlarge, ml.r6g.large, ml.r6g.xlarge, ml.r6g.2xlarge, ml.r6g.4xlarge, ml.r6g.8xlarge, ml.r6g.12xlarge, ml.r6g.16xlarge, ml.r6gd.large, ml.r6gd.xlarge, ml.r6gd.2xlarge, ml.r6gd.4xlarge, ml.r6gd.8xlarge, ml.r6gd.12xlarge, ml.r6gd.16xlarge, ml.p4de.24xlarge, ml.trn1.2xlarge, ml.trn1.32xlarge, ml.trn1n.32xlarge, ml.inf2.xlarge, ml.inf2.8xlarge, ml.inf2.24xlarge, ml.inf2.48xlarge, ml.inf2e.32xlarge, ml.p5.48xlarge, ml.m7i.large, ml.m7i.xlarge, ml.m7i.2xlarge, ml.m7i.4xlarge, ml.m7i.8xlarge, ml.m7i.12xlarge, ml.m7i.16xlarge, ml.m7i.24xlarge, ml.m7i.48xlarge, ml.c7i.large, ml.c7i.xlarge, ml.c7i.2xlarge, ml.c7i.4xlarge, ml.c7i.8xlarge, ml.c7i.12xlarge, ml.c7i.16xlarge, ml.c7i.24xlarge, ml.c7i.48xlarge, ml.r7i.large, ml.r7i.xlarge, ml.r7i.2xlarge, ml.r7i.4xlarge, ml.r7i.8xlarge, ml.r7i.12xlarge, ml.r7i.16xlarge, ml.r7i.24xlarge, ml.r7i.48xlarge]

transform_instance_type = "ml.m4.xlarge"

deployment_instance_type = ParameterString(name="ProcessingInstanceType", default_value="ml.t2.medium")
