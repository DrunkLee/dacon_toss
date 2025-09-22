import pyarrow.parquet as pq
from pipeline import Pipeline

train_path = "./data/train.parquet"
test_path  = "./data/test.parquet"

pf = pq.ParquetFile(train_path)
feature_cols = [c for c in pf.schema.names if c != "seq"]