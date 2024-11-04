
import json
from hs_pipeline import get_pipeline
from hs_pipeline.conf import role_arn

pipeline = get_pipeline()

definition = json.loads(pipeline.definition())
pipeline.upsert(role_arn=role_arn)
execution = pipeline.start()
execution.wait()
