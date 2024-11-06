
from sagemaker.workflow.parameters import ParameterInteger, ParameterString, ParameterFloat

# model performance step parameters
accuracy_mse_threshold   = ParameterFloat(name="AccuracyMseThreshold", default_value=0.75)