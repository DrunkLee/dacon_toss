import pyarrow.parquet as pq
from dataloader import DataLoader

train_path = "./data/train.parquet"
test_path  = "./data/test.parquet"

loader = DataLoader()

pf = pq.ParquetFile(train_path)
feature_cols = [c for c in pf.schema.names if c != "seq"]
df = loader.load_raw_parquet(train_path, feature_cols=feature_cols)
df.head(10)