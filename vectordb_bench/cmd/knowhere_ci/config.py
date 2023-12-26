from vectordb_bench.backend.cases import CaseType

drop_old = True

task_label = "knowhere_ci"

case_ids: list[CaseType] = [
    CaseType.PerformanceCohereInternal,
    CaseType.PerformanceOpenAIInternal,
    CaseType.PerformanceGIST768Internal,
    CaseType.PerformanceText2imgInternal,
]

build_threads = 8
search_threads = 8


def get_ivfflat_params():
    return {
        "build": {
            "nlist": 1024,
        },
        "search": {
            "nprobe": [8, 16, 32, 64, 128],
        },
    }


def get_hnsw_params():
    return {
        "build": {
            "efConstruction": 360,
            "M": 30,
        },
        "search": {
            "ef": [100, 200, 400],
        },
    }


def get_diskann_params(num_rows: int, dim: int) -> dict:
    pq_code_budget_gb = num_rows * dim * 4 * 0.125 / 1024 / 1024 / 1024
    search_cache_budget_gb = num_rows * dim * 4 * 0.125 / 1024 / 1024 / 1024
    diskann_params = {
        "build": {
            "max_degree": 56,
            "build_dram_budget_gb": 16,
            "search_list_size": 128,
            "pq_code_budget_gb": pq_code_budget_gb,
            "search_cache_budget_gb": search_cache_budget_gb,
        },
        "search": {
            "search_list_size": [100, 200, 400],
        },
    }
    return diskann_params
