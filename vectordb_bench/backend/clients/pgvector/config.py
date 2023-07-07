from pydantic import BaseModel, SecretStr
from ..api import DBConfig, DBCaseConfig, MetricType

POSTGRE_URL_PLACEHOLDER = "postgresql://%s:%s@%s/%s"
INDEX_TYPE = "ivfflat"

class PgVectorConfig(DBConfig):
    user_name: SecretStr
    password: SecretStr
    url: SecretStr
    db_name: str

    def to_dict(self) -> dict:
        user_str = self.user_name.get_secret_value()
        pwd_str = self.password.get_secret_value()
        url_str = self.url.get_secret_value()
        return {
            "url" : POSTGRE_URL_PLACEHOLDER%(user_str, pwd_str, url_str, db_name)
        }

class PgVectorIndexConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None
    lists: int | None = 10
    probes: int | None = 1

    def parse_metric(self) -> str: 
        if self.metric_type == MetricType.L2:
            return "vector_l2_ops"
        elif self.metric_type == MetricType.IP:
            return "vector_ip_ops"
        return "vector_cosine_ops"
    
    def parse_metric_fun_str(self) -> str: 
        if self.metric_type == MetricType.L2:
            return "l2_distance"
        elif self.metric_type == MetricType.IP:
            return "max_inner_product"
        return "cosine_distance"

    def index_param(self) -> dict:
        return {
            "postgresql_using" : INDEX_TYPE,
            "postgresql_with" : {'lists': self.lists},
            "postgresql_ops": self.parse_metric()
        }
    
    def search_param(self) -> dict:
        return {
            "probes" : probes,
            "metric_fun" : self.parse_metric_fun_str()
        }

    