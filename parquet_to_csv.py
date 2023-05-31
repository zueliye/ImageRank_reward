import pyarrow as pa
import pyarrow.parquet as pq
df = pq.read_table('train-00000-of-00001-5088566feaadd3d9.parquet').to_pandas()
df.to_csv('train_rank.csv')