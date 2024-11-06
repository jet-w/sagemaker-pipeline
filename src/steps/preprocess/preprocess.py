import os
import pandas as pd

#####################################################
## Will be replaced using code in the future
base_dir        = "/opt/ml/processing"

if __name__ == "__main__":
    df = None
    for p, _, files in os.walk(f"{base_dir}/input"):
        df = pd.concat(list(map(lambda x: pd.read_csv(os.path.join(p, x)), files)))
    rows = df.shape[0]
    
    # Sample data
    training = df.sample(frac=0.8, random_state=42)
    test = df.drop(training.index)

    training.to_csv(f"{base_dir}/train/train.csv", header=True, index=False)
    test.to_csv(f"{base_dir}/test/test.csv", header=True, index=False)