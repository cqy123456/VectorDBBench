## Env

### nas
All test datasets are from NAS.

```sh
mkdir -p /home/nas
mount -t nfs 172.168.70.248:/zilliz/milvus /home/nas
```

### git clone vdbb
Note that, VDBBench is a private repo.

```sh
git clone -b ci_for_knowhere git@github.com:zilliztech/VDBBench.git
```

### install vdbb

```sh
cd VDBBench
pip install -e ".[test]"
```

## Run

config file location: "vectordb_bench/cmd/knowhere_ci/config.py"

```
cmd_bench
```

## Get Results

use `Python`
```python
from vectordb_bench.cmd.knowhere_ci.run import get_knowhere_test_results

data = get_knowhere_test_results()

print(data)
"""
{'search_vps': 27493.0491, 'search_recall': 0.0085, 'build_time': 4.4298, 'dataset_name': 'Cohere', 'data_rows': 1000000, 'data_dim': 768, 'metric_type': 'COSINE', 'index_type': 'IVFFLAT', 'build_params': '{"nlist": 1024}', 'search_params': '{"nprobe": 8}'}
{'search_vps': 18221.4491, 'search_recall': 0.0092, 'build_time': 4.4298, 'dataset_name': 'Cohere', 'data_rows': 1000000, 'data_dim': 768, 'metric_type': 'COSINE', 'index_type': 'IVFFLAT', 'build_params': '{"nlist": 1024}', 'search_params': '{"nprobe": 16}'}
{'search_vps': 10369.468, 'search_recall': 0.0096, 'build_time': 3.1056, 'dataset_name': 'Cohere', 'data_rows': 1000000, 'data_dim': 768, 'metric_type': 'COSINE', 'index_type': 'HNSW', 'build_params': '{"efConstruction": 360, "M": 30}', 'search_params': '{"ef": 100}'}
{'search_vps': 6401.8017, 'search_recall': 0.0096, 'build_time': 3.1056, 'dataset_name': 'Cohere', 'data_rows': 1000000, 'data_dim': 768, 'metric_type': 'COSINE', 'index_type': 'HNSW', 'build_params': '{"efConstruction": 360, "M": 30}', 'search_params': '{"ef": 200}'}
"""
```