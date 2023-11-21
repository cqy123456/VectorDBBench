from pydantic import BaseModel
from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients.milvus.config import get_col_ids_by_case
from vectordb_bench.models import CaseConfig
from ..api import DBConfig, DBCaseConfig, MetricType, TestType
import json


class KnowhereCloudConfig(DBConfig):
    test_type: str = TestType.LIBRARY.value
    index_type: str = "HNSW"
    config: str = (
        '"M": 30, "efConstruction": 360, "ef": 100, "nlist": 1024, "nprobe": 64'
    )

    def to_dict(self) -> dict:
        return {
            "index_type": self.index_type,
            "config": self.config,
        }

    @property
    def config_json(self):
        config = json.loads(f"{{{self.config}}}")
        config["index_type"] = self.index_type
        return config


class KnowhereCloudIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    caseConfig: CaseConfig = CaseConfig(case_id=CaseType.Custom, custom_case={})

    def parse_metric(self) -> str:
        return self.metric_type.value

    def index_param(self) -> dict:
        return {"metric_type": self.parse_metric()}

    def search_param(self) -> str:
        efConstruction = 8
        col_ids = get_col_ids_by_case(self.caseConfig)
        for col_id in col_ids:
            if col_id == 0:
                efConstruction += 1
            else:
                efConstruction += 2 << (col_id - 1)
        return {"metric_type": self.parse_metric(), "efConstruction": efConstruction}
