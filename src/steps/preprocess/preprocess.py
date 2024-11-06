import os
import pandas as pd
import logging
#####################################################
## Will be replaced using code in the future
base_dir        = "/opt/ml/processing"

if __name__ == "__main__":
    df = None
    logging.info("Start preprocess")
    for p, _, files in os.walk(f"{base_dir}/input"):
        logging.info(p)
        logging.info(files)
        df = pd.concat(
                list(
                    map(
                        lambda x: pd.read_csv(os.path.join(p, x)), 
                        filter(lambda f: f.lower().endswith(".csv"), files)
                    )
                )
             )
    rows = df.shape[0]
    logging.info(f"Start preprocess: {rows} records")
    # Sample data
    training = df.sample(frac=0.8, random_state=42)
    test = df.drop(training.index)

    training.to_csv(f"{base_dir}/train/train.csv", header=True, index=False)
    test.to_csv(f"{base_dir}/test/test.csv", header=True, index=False)