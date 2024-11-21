from aws_cdk import (
    Stack,
    aws_iam as iam,
    aws_ec2 as ec2,
    aws_sagemaker as sagemaker,
    CfnOutput,
    aws_lambda as _lambda,
    aws_logs
)
import json
#from aws_cdk import aws_lambda_python_alpha as _alambda
from constructs import Construct
from aws_cdk.custom_resources import Provider
import aws_cdk as core
from constructs import Construct
from cdk_nag import NagSuppressions

class HSSagemakerStudioSetupStack(Stack):
    # Standard definition for CDK stack
    def __init__(self, scope: Construct, construct_id: str,  **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)
        
        filename = "project_config.json"
        variables = json.load(open(filename))

        sagemaker_domain_name = variables["SageMaker"]["Domain"]["Name"]
        user_profiles     = variables["SageMaker"]["UserProfiles"]
        execution_role    = variables["SageMaker"]["IAMRole"]["Name"]
        vpc_name          = variables["SageMaker"]["VPC"]["Name"]
        log_group_name    = variables["SageMaker"]["VPC"]["LogGroup"]["Name"]
        log_group_iam_role= variables["SageMaker"]["VPC"]["IAMRole"]["Name"]
        flow_log_name     = variables["SageMaker"]["VPC"]["FlowLog"]["Name"]
        # Create Studio Role
        role = iam.Role(
            self,
            "Studio Role",
            role_name = execution_role,
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"), #When you are moving to production follow least privilage access. Use custom policies instead of managed access policies
            ],
            inline_policies={
                "CustomRules": iam.PolicyDocument(
                    statements=[
                        iam.PolicyStatement(
                            actions=[
                                "codewhisperer:GenerateRecommendations*",
                            ],
                            resources=["*"],
                        )
                    ]
                )
            },
        )

        vpc = ec2.Vpc(self, "VPC", vpc_name=vpc_name)

        self.public_subnet_ids = [
            public_subnet.subnet_id for public_subnet in vpc.public_subnets
        ]

        flow_log_group = aws_logs.LogGroup(self, "vpcFlowLogGroup", log_group_name=log_group_name)

        flow_log_role = iam.Role(self, 
                                "vpcFLowLogRole",
                                assumed_by=iam.ServicePrincipal("vpc-flow-logs.amazonaws.com"),
                                role_name=log_group_iam_role
                                )

        ec2.FlowLog(self, "FlowLog",
            resource_type=ec2.FlowLogResourceType.from_vpc(vpc),
            flow_log_name= flow_log_name,
            destination=ec2.FlowLogDestination.to_cloud_watch_logs(flow_log_group, flow_log_role)
        )
        
        # Create domain with IAM auth, role created above, VPC created above and subnets created above
        domain = sagemaker.CfnDomain(
            self,
            "Domain",
            auth_mode="IAM",
            default_user_settings=sagemaker.CfnDomain.UserSettingsProperty(
                execution_role=role.role_arn,
            ),
            #domain_name=f"Studio-{stack_name}",
            domain_name=sagemaker_domain_name,
            subnet_ids=self.public_subnet_ids,
            vpc_id=vpc.vpc_id,
        )
        # Create User Profile For Sagemaker Domain
        if user_profiles:
            for user in user_profiles:
                sagemaker.CfnUserProfile(
                    self,
                    f"User-{user}",
                    domain_id=domain.attr_domain_id,
                    user_profile_name=user,
                    user_settings=sagemaker.CfnUserProfile.UserSettingsProperty(
                        execution_role=role.role_arn,
                    ),
                )

        CfnOutput(self, "domain_id", value=domain.attr_domain_id)
        
        ## CDK NAG suppression
        NagSuppressions.add_resource_suppressions(role,
                            suppressions=[{
                                            "id": "AwsSolutions-IAM4",
                                            "reason": "Sagemaker Notebook policies need to be broad to allow access to ",
                                            },{
                                            "id": "AwsSolutions-IAM5",
                                            "reason": "SageMaker Studio Role requires access to all indicies",
                                            }
                                        ],
                            apply_to_children=True)
        NagSuppressions.add_stack_suppressions(self, [
                                {
                                    "id": 'AwsSolutions-IAM4',
                                    "reason": 'Lambda execution policy for custom resources created by higher level CDK constructs',
                                    "appliesTo": [
                                            'Policy::arn:<AWS::Partition>:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole',
                                        ],
                                }])