#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(__file__))

import aws_cdk as cdk

#from cdk_test.cdk_test_stack import DockerBuildStack
from stacks.hs_sagemaker_studio_setup_stack import HSSagemakerStudioSetupStack
#from stacks.hs_sagemaker_pipeline_stack import HSSagemakerPipelineStack
from stacks.hs_code_pipeline_stack import Pipeline


app = cdk.App()

hs_sagemaker_studio_stack = HSSagemakerStudioSetupStack(app, "HSMLOpsSagemakerStudioSetupStack",
    # If you don't specify 'env', this stack will be environment-agnostic.
    # Account/Region-dependent features and context lookups will not work,
    # but a single synthesized template can be deployed anywhere.

    # Uncomment the next line to specialize this stack for the AWS Account
    # and Region that are implied by the current CLI configuration.
    
    # Uncomment the next line if you know exactly what Account and Region you
    # want to deploy the stack to. */
    env=cdk.Environment(account='654654179472', region='us-east-1'),
    #env=cdk.Environment(account=os.getenv('CDK_DEFAULT_ACCOUNT'), region=os.getenv('CDK_DEFAULT_REGION')),
    # For more information, see https://docs.aws.amazon.com/cdk/latest/guide/environments.html
)

#hs_sagemaker_pipeline_stack = HSSagemakerPipelineStack(app, "HSMLOpsHSSagemakerPipelineStack",
#    env=cdk.Environment(account='654654179472', region='us-east-1'),
#)
#hs_sagemaker_pipeline_stack.add_dependency(hs_sagemaker_studio_stack)
hs_code_pipeline = Pipeline(app, "mlops-code-pipelines", branch="master", env=cdk.Environment(account='654654179472', region='us-east-1'),)
hs_code_pipeline.add_dependency(hs_sagemaker_studio_stack)

app.synth()
