
import json
import os
import sys
sys.path.append(os.path.dirname(__file__))
from hs_pipeline import get_pipeline
from etc import role_arn

if __name__ == "__main__":
    pipeline = get_pipeline()
    #definition = json.loads(pipeline.definition())
    pipeline.upsert(role_arn=role_arn)
    execution = pipeline.start()
    execution.wait()
