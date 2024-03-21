from pydantic import BaseModel
from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients.milvus.config import get_col_ids_by_case
from vectordb_bench.models import CaseConfig
from ..api import DBConfig, DBCaseConfig, MetricType, TestType
import json


class KnowhereCloudConfig(DBConfig):
    test_type: str = TestType.LIBRARY.value
    index_type: str = "HNSW"
    build_threads: int = 2
    search_threads: int = 2
    with_cardinal: int = 0 # 0 means false
    config: str = (
        '"M": 30, "efConstruction": 360, "ef": 100, "nlist": 1024, "nprobe": 64'
    )

    def to_dict(self) -> dict:
        return {
            "index_type": self.index_type,
            "config": self.config,
            "search_threads": self.search_threads,
            "build_threads": self.build_threads,
            "with_cardinal": self.with_cardinal,
        }

    @property
    def config_json(self):
        config = json.loads(f"{{{self.config}}}")
        config["index_type"] = self.index_type
        return config


class KnowhereCloudIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    caseConfig: CaseConfig = CaseConfig(case_id=CaseType.Custom, custom_case={})
    nprobe: int | None = None
    ef: int | None = None
    search_list_size: int | None = None

    def parse_metric(self) -> str:
        return self.metric_type.value

    def index_param(self) -> dict:
        return {"metric_type": self.parse_metric()}

    def search_param(self) -> str:
        params = {"metric_type": self.parse_metric()}
        if self.nprobe != None:
            params["nprobe"] = self.nprobe
        if self.ef != None:
            params["ef"] = self.ef
        if self.search_list_size != None:
            params["search_list_size"] = self.search_list_size
        return params
