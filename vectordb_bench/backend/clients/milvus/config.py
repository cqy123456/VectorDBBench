from pydantic import BaseModel, SecretStr
from vectordb_bench.backend.cases import CaseType
from vectordb_bench.models import CaseConfig
from ..api import DBConfig, DBCaseConfig, MetricType, IndexType


class MilvusConfig(DBConfig):
    uri: SecretStr = "http://localhost:19530"

    def to_dict(self) -> dict:
        return {"uri": self.uri.get_secret_value()}


class MilvusIndexConfig(BaseModel):
    """Base config for milvus"""

    index: IndexType
    metric_type: MetricType | None = None

    def parse_metric(self) -> str:
        if not self.metric_type:
            return ""

        # if self.metric_type == MetricType.COSINE:
        #     return MetricType.L2.value
        return self.metric_type.value


class AutoIndexConfig(MilvusIndexConfig, DBCaseConfig):
    index: IndexType = IndexType.AUTOINDEX

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
        }


def get_col_ids_by_case(caseConfig: CaseConfig):
    case = caseConfig.case_id.case_cls(caseConfig.custom_case)
    filter_rate = case.filter_rate
    if filter_rate is None or filter_rate == 0.0:
        return [0]
    elif case.case_id == CaseType.CustomIntFilter:
        return [0, 6]
    elif case.case_id == CaseType.CustomCategoryFilter:
        return [case.category_column_idx + 1]
    elif case.case_id == CaseType.CustomAndFilter:
        return [
            category_column_idx + 1
            for category_column_idx in case.category_column_idxes
        ]
    elif case.case_id == CaseType.CustomOrFilter:
        if len(case.category_column_idxes) == 1:
            return [
                category_column_idx + 1
                for category_column_idx in case.category_column_idxes
            ]
        else:
            return [
                0,
                *[
                    category_column_idx + 1
                    for category_column_idx in case.category_column_idxes
                ],
            ]
    return [0]


class HNSWConfig(MilvusIndexConfig, DBCaseConfig):
    M: int = 30
    efConstruction: int = 360
    ef: int = 100
    index: IndexType = IndexType.HNSW
    caseConfig: CaseConfig = CaseConfig(case_id=CaseType.Performance768D1M, custom_case={})

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {"M": self.M, "efConstruction": self.efConstruction},
        }

    def search_param(self) -> dict:
        efConstruction = 8
        col_ids = get_col_ids_by_case(self.caseConfig)
        for col_id in col_ids:
            if col_id == 0:
                efConstruction += 1
            else:
                efConstruction += 2 << (col_id - 1)
        return {
            "metric_type": self.parse_metric(),
            "params": {"ef": self.ef, "efConstruction": efConstruction},
        }


class DISKANNConfig(MilvusIndexConfig, DBCaseConfig):
    search_list: int = 100
    index: IndexType = IndexType.DISKANN

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {"search_list": self.search_list},
        }


class IVFFlatConfig(MilvusIndexConfig, DBCaseConfig):
    nlist: int
    nprobe: int | None = None
    index: IndexType = IndexType.IVFFlat

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {"nlist": self.nlist},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {"nprobe": self.nprobe},
        }


class FLATConfig(MilvusIndexConfig, DBCaseConfig):
    index: IndexType = IndexType.Flat

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {},
        }


_milvus_case_config = {
    IndexType.AUTOINDEX: AutoIndexConfig,
    IndexType.HNSW: HNSWConfig,
    IndexType.DISKANN: DISKANNConfig,
    IndexType.IVFFlat: IVFFlatConfig,
    IndexType.Flat: FLATConfig,
}
