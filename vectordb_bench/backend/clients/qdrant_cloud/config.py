from pydantic import BaseModel, SecretStr

from ..api import DBConfig, DBCaseConfig, MetricType
from qdrant_client.models import Distance


class QdrantConfig(DBConfig):
    index_type: SecretStr
    search_params: SecretStr

    def to_dict(self) -> dict:
        return {
            "index_type": self.index_type.get_secret_value(),
            "search_params": int(self.search_params.get_secret_value())
        }

class QdrantIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.L2:
            return Distance.EUCLID
        elif self.metric_type == MetricType.IP:
            return Distance.DOT
        return Distance.COSINE

    def index_param(self) -> dict:
        params = {"distance": self.parse_metric()}
        return params

    def search_param(self) -> dict:
        return {}