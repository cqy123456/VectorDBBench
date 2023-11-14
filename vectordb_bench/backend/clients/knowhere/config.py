from pydantic import BaseModel
from ..api import DBConfig, DBCaseConfig, MetricType, TestType
import json


class KnowhereConfig(DBConfig):
    test_type: str = TestType.LIBRARY.value
    index_type: str = "HNSW"
    config: str = (
        '"M": 24, "efConstruction": 100, "ef": 100, "nlist": 1024, "nprobe": 64'
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


class KnowhereIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        return self.metric_type.value

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric()
        }
        
    def search_param(self) -> str:
        return self.index_param()
