import typing
import logging
from enum import Enum, auto

from vectordb_bench import config
from vectordb_bench.base import BaseModel

from .dataset import (
    Dataset,
    DatasetManager,
    ScalarDatasetLabel,
    category_num_to_column_name,
)


log = logging.getLogger(__name__)

Case = typing.TypeVar("Case")


class CaseType(Enum):
    """
    Example:
        >>> case_cls = CaseType.CapacityDim128.case_cls
        >>> assert c is not None
        >>> CaseType.CapacityDim128.case_name
        "Capacity Test (128 Dim Repeated)"
    """

    CapacityDim128 = 1
    CapacityDim960 = 2

    Performance768D100M = 3
    Performance768D10M = 4
    Performance768D1M = 5

    Performance768D10M1P = 6
    Performance768D1M1P = 7
    Performance768D10M99P = 8
    Performance768D1M99P = 9
    PerformanceGlove200 = "PerformanceGlove200"
    PerformanceLastFM = "PerformanceLastFM"
    PerformanceGIST768Internal = "PerformanceGIST768Internal"
    PerformanceText2imgInternal = "PerformanceText2imgInternal"
    PerformanceCohereInternal = "PerformanceCohereInternal"
    PerformanceOpenAIInternal = "PerformanceOpenAIInternal"

    Performance1536D500K = 10
    Performance1536D5M = 11

    Performance1536D500K1P = 12
    Performance1536D5M1P = 13
    Performance1536D500K99P = 14
    Performance1536D5M99P = 15

    Custom = 100
    CustomIntFilter = "CustomIntFilter"
    CustomCategoryFilter = "CustomCategoryFilter"
    CustomAndFilter = "CustomAndFilter"
    CustomOrFilter = "CustomOrFilter"

    def case_cls(self, custom_configs: dict | None = None) -> Case:
        return type2case.get(self)(
            **(custom_configs if custom_configs is not None else {})
        )

    def case_name(self, custom_configs: dict | None = None) -> str:
        c = self.case_cls(custom_configs)
        if c is not None:
            return c.name
        raise ValueError("Case unsupported")

    def case_description(self, custom_configs: dict | None = None) -> str:
        c = self.case_cls(custom_configs)
        if c is not None:
            return c.description
        raise ValueError("Case unsupported")


class CaseLabel(Enum):
    Load = auto()
    Performance = auto()


class Case(BaseModel):
    """Undifined case

    Fields:
        case_id(CaseType): default 9 case type plus one custom cases.
        label(CaseLabel): performance or load.
        dataset(DataSet): dataset for this case runner.
        filter_rate(float | None): one of 99% | 1% | None
        filters(dict | None): filters for search
    """

    case_id: CaseType
    label: CaseLabel
    name: str = ""
    description: str = ""
    dataset: DatasetManager

    load_timeout: float | int
    optimize_timeout: float | int | None

    filter_rate: float | None
    with_category_column: bool = False

    @property
    def filters(self) -> dict | None:
        if self.filter_rate is not None:
            ID = round(self.filter_rate * self.dataset.data.size)
            return {
                "metadata": f">={ID}",
                "id": ID,
            }

        return None

    def get_ground_truth_file(self) -> str:
        filter_rate = self.filter_rate
        if filter_rate is None or filter == 0.0:
            file_name = "neighbors.parquet"
        elif filter_rate == 0.01:
            file_name = "neighbors_head_1p.parquet"
        elif filter_rate == 0.99:
            file_name = "neighbors_tail_1p.parquet"
        else:
            raise ValueError(f"Filters not supported: {filter_rate}")
        return file_name


class CapacityCase(Case, BaseModel):
    label: CaseLabel = CaseLabel.Load
    filter_rate: float | None = None
    load_timeout: float | int = config.CAPACITY_TIMEOUT_IN_SECONDS
    optimize_timeout: float | int | None = None


class PerformanceCase(Case, BaseModel):
    label: CaseLabel = CaseLabel.Performance
    filter_rate: float | None = None
    load_timeout: float | int = config.LOAD_TIMEOUT_DEFAULT
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_DEFAULT


class CapacityDim960(CapacityCase):
    case_id: CaseType = CaseType.CapacityDim960
    dataset: DatasetManager = Dataset.GIST.manager(100_000)
    name: str = "Capacity Test (960 Dim Repeated)"
    description: str = """This case tests the vector database's loading capacity by repeatedly inserting large-dimension vectors (GIST 100K vectors, <b>960 dimensions</b>) until it is fully loaded.
Number of inserted vectors will be reported."""


class CapacityDim128(CapacityCase):
    case_id: CaseType = CaseType.CapacityDim128
    dataset: DatasetManager = Dataset.SIFT.manager(500_000)
    name: str = "Capacity Test (128 Dim Repeated)"
    description: str = """This case tests the vector database's loading capacity by repeatedly inserting small-dimension vectors (SIFT 100K vectors, <b>128 dimensions</b>) until it is fully loaded.
Number of inserted vectors will be reported."""


class PerformanceGlove200(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceGlove200
    dataset: DatasetManager = Dataset.GLOVE_200.manager()
    name: str = "GloVe 200 (1M+, 200 Dim, Cosine)"
    description: str = """Glove 200, dim=200, n=1,183,514"""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_1M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_1M


class PerformanceLastFM(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceLastFM
    dataset: DatasetManager = Dataset.LASTFM.manager()
    name: str = "Last.FM (~300K, 65 Dim, Cosine)"
    description: str = """LASTFM, dim=65, n=292,385"""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_1M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_1M


class PerformanceGIST768Internal(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceGIST768Internal
    dataset: DatasetManager = Dataset.ScalarGIST768.manager()
    with_category_column: bool = False
    name: str = "GIST 768 (1M, 768 Dim, L2)"
    description: str = """GIST, dim=768, n=1,000,000, Internal"""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_1M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_1M


class PerformanceText2imgInternal(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceText2imgInternal
    dataset: DatasetManager = Dataset.ScalarText2img.manager()
    with_category_column: bool = False
    name: str = "Text2img (5M, 200 Dim, IP)"
    description: str = """Text2img, dim=200, n=5,000,000, Internal"""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_1M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_1M


class PerformanceCohereInternal(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceCohereInternal
    dataset: DatasetManager = Dataset.ScalarCohere.manager()
    with_category_column: bool = False
    name: str = "Cohere (1M, 768 Dim, Cosine)"
    description: str = """Cohere, dim=768, n=1,000,000, Internal"""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_1M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_1M


class PerformanceOpenAIInternal(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceOpenAIInternal
    dataset: DatasetManager = Dataset.ScalarOpenAI.manager()
    with_category_column: bool = False
    name: str = "OpenAI (500K, 1536 Dim, Cosine)"
    description: str = """OpenAI, dim=1536, n=500,000, Internal"""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_1M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_1M


class Performance768D10M(PerformanceCase):
    case_id: CaseType = CaseType.Performance768D10M
    dataset: DatasetManager = Dataset.COHERE.manager(10_000_000)
    name: str = "Search Performance Test (10M Dataset, 768 Dim)"
    description: str = """This case tests the search performance of a vector database with a large dataset (<b>Cohere 10M vectors</b>, 768 dimensions) at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_10M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_10M


class Performance768D1M(PerformanceCase):
    case_id: CaseType = CaseType.Performance768D1M
    dataset: DatasetManager = Dataset.COHERE.manager(1_000_000)
    name: str = "Search Performance Test (1M Dataset, 768 Dim)"
    description: str = """This case tests the search performance of a vector database with a medium dataset (<b>Cohere 1M vectors</b>, 768 dimensions) at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_1M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_1M


class Performance768D10M1P(PerformanceCase):
    case_id: CaseType = CaseType.Performance768D10M1P
    filter_rate: float | int | None = 0.01
    dataset: DatasetManager = Dataset.COHERE.manager(10_000_000)
    name: str = "Filtering Search Performance Test (10M Dataset, 768 Dim, Filter 1%)"
    description: str = """This case tests the search performance of a vector database with a large dataset (<b>Cohere 10M vectors</b>, 768 dimensions) under a low filtering rate (<b>1% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_10M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_10M


class Performance768D1M1P(PerformanceCase):
    case_id: CaseType = CaseType.Performance768D1M1P
    filter_rate: float | int | None = 0.01
    dataset: DatasetManager = Dataset.COHERE.manager(1_000_000)
    name: str = "Filtering Search Performance Test (1M Dataset, 768 Dim, Filter 1%)"
    description: str = """This case tests the search performance of a vector database with a medium dataset (<b>Cohere 1M vectors</b>, 768 dimensions) under a low filtering rate (<b>1% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_1M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_1M


class Performance768D10M99P(PerformanceCase):
    case_id: CaseType = CaseType.Performance768D10M99P
    filter_rate: float | int | None = 0.99
    dataset: DatasetManager = Dataset.COHERE.manager(10_000_000)
    name: str = "Filtering Search Performance Test (10M Dataset, 768 Dim, Filter 99%)"
    description: str = """This case tests the search performance of a vector database with a large dataset (<b>Cohere 10M vectors</b>, 768 dimensions) under a high filtering rate (<b>99% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_10M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_10M


class Performance768D1M99P(PerformanceCase):
    case_id: CaseType = CaseType.Performance768D1M99P
    filter_rate: float | int | None = 0.99
    dataset: DatasetManager = Dataset.COHERE.manager(1_000_000)
    name: str = "Filtering Search Performance Test (1M Dataset, 768 Dim, Filter 99%)"
    description: str = """This case tests the search performance of a vector database with a medium dataset (<b>Cohere 1M vectors</b>, 768 dimensions) under a high filtering rate (<b>99% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_1M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_1M


class Performance768D100M(PerformanceCase):
    case_id: CaseType = CaseType.Performance768D100M
    filter_rate: float | int | None = None
    dataset: DatasetManager = Dataset.LAION.manager(100_000_000)
    name: str = "Search Performance Test (100M Dataset, 768 Dim)"
    description: str = """This case tests the search performance of a vector database with a large 100M dataset (<b>LAION 100M vectors</b>, 768 dimensions), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_768D_100M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_768D_100M


class Performance1536D500K(PerformanceCase):
    case_id: CaseType = CaseType.Performance1536D500K
    filter_rate: float | int | None = None
    dataset: DatasetManager = Dataset.OPENAI.manager(500_000)
    name: str = "Search Performance Test (500K Dataset, 1536 Dim)"
    description: str = """This case tests the search performance of a vector database with a medium 500K dataset (<b>OpenAI 500K vectors</b>, 1536 dimensions), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_1536D_500K
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_1536D_500K


class Performance1536D5M(PerformanceCase):
    case_id: CaseType = CaseType.Performance1536D5M
    filter_rate: float | int | None = None
    dataset: DatasetManager = Dataset.OPENAI.manager(5_000_000)
    name: str = "Search Performance Test (5M Dataset, 1536 Dim)"
    description: str = """This case tests the search performance of a vector database with a medium 5M dataset (<b>OpenAI 5M vectors</b>, 1536 dimensions), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_1536D_5M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_1536D_5M


class Performance1536D500K1P(PerformanceCase):
    case_id: CaseType = CaseType.Performance1536D500K1P
    filter_rate: float | int | None = 0.01
    dataset: DatasetManager = Dataset.OPENAI.manager(500_000)
    name: str = "Filtering Search Performance Test (500K Dataset, 1536 Dim, Filter 1%)"
    description: str = """This case tests the search performance of a vector database with a large dataset (<b>OpenAI 500K vectors</b>, 1536 dimensions) under a low filtering rate (<b>1% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_1536D_500K
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_1536D_500K


class Performance1536D5M1P(PerformanceCase):
    case_id: CaseType = CaseType.Performance1536D5M1P
    filter_rate: float | int | None = 0.01
    dataset: DatasetManager = Dataset.OPENAI.manager(5_000_000)
    name: str = "Filtering Search Performance Test (5M Dataset, 1536 Dim, Filter 1%)"
    description: str = """This case tests the search performance of a vector database with a large dataset (<b>OpenAI 5M vectors</b>, 1536 dimensions) under a low filtering rate (<b>1% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_1536D_5M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_1536D_5M


class Performance1536D500K99P(PerformanceCase):
    case_id: CaseType = CaseType.Performance1536D500K99P
    filter_rate: float | int | None = 0.99
    dataset: DatasetManager = Dataset.OPENAI.manager(500_000)
    name: str = "Filtering Search Performance Test (500K Dataset, 1536 Dim, Filter 99%)"
    description: str = """This case tests the search performance of a vector database with a medium dataset (<b>OpenAI 500K vectors</b>, 1536 dimensions) under a high filtering rate (<b>99% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_1536D_500K
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_1536D_500K


class Performance1536D5M99P(PerformanceCase):
    case_id: CaseType = CaseType.Performance1536D5M99P
    filter_rate: float | int | None = 0.99
    dataset: DatasetManager = Dataset.OPENAI.manager(5_000_000)
    name: str = "Filtering Search Performance Test (5M Dataset, 1536 Dim, Filter 99%)"
    description: str = """This case tests the search performance of a vector database with a medium dataset (<b>OpenAI 5M vectors</b>, 1536 dimensions) under a high filtering rate (<b>99% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_1536D_5M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_1536D_5M


class CustomFilter(PerformanceCase):
    dataset_label: ScalarDatasetLabel
    with_category_column: bool = True


class CustomIntFilter(CustomFilter):
    case_id: CaseType = CaseType.CustomIntFilter
    filter_rate: float | None = None

    def __init__(
        self,
        dataset_label: ScalarDatasetLabel | str,
        filter_rate: float | None,
        **kwargs,
    ):
        dataset = ScalarDatasetLabel(dataset_label).get_dataset()
        name = (
            f"int-{int(filter_rate * 100)}p"
            if filter_rate is not None
            else f"{filter_rate}"
        )
        description = f"{dataset.data.name} - {name}"
        super().__init__(
            dataset_label=dataset_label,
            dataset=dataset,
            filter_rate=filter_rate,
            name=name,
            description=description,
            **kwargs,
        )

    def get_ground_truth_file(self) -> str:
        filter_rate = self.filter_rate
        return (
            f"neighbors_id_filter_{int(filter_rate * 100)}p.parquet"
            if filter_rate is not None and filter_rate > 0
            else "neighbors.parquet"
        )


def category_num_to_category_column_idx(
    dataset: DatasetManager, category_num: int
) -> int:
    return dataset.data.category_nums.index(category_num)


class CustomCategoryFilter(CustomFilter):
    case_id: CaseType = CaseType.CustomCategoryFilter
    filter_rate: float
    category_num: int
    category_column_idx: int

    def __init__(
        self, dataset_label: ScalarDatasetLabel | str, category_num: int, **kwargs
    ):
        filter_rate = 1.0 - 1.0 / category_num
        dataset = ScalarDatasetLabel(dataset_label).get_dataset()
        category_column_idx = category_num_to_category_column_idx(dataset, category_num)
        name = f"{category_num_to_column_name(category_num)}"
        description = f"{dataset.data.name} - {name}"
        super().__init__(
            dataset_label=dataset_label,
            dataset=dataset,
            filter_rate=filter_rate,
            category_num=category_num,
            category_column_idx=category_column_idx,
            name=name,
            description=description,
            **kwargs,
        )

    def get_ground_truth_file(self) -> str:
        return f"neighbors_{self.name.replace('-', '_')}_filter.parquet"


class CustomAndFilter(CustomFilter):
    case_id: CaseType = CaseType.CustomAndFilter
    filter_rate: float
    category_nums: list[int]
    category_column_idxes: list[int]

    def __init__(
        self,
        dataset_label: ScalarDatasetLabel | str,
        category_nums: list[int],
        **kwargs,
    ):
        r = 1
        for category_num in category_nums:
            r *= category_num
        filter_rate = 1.0 - 1.0 / r
        dataset = ScalarDatasetLabel(dataset_label).get_dataset()
        category_column_idxes = [
            category_num_to_category_column_idx(dataset, category_num)
            for category_num in category_nums
        ]
        category_nums.sort()
        name = f"{len(category_nums)}_and_{'_'.join([f'{category_num}' for category_num in category_nums])}"
        description = f"{dataset.data.name} - {name}"
        super().__init__(
            dataset_label=dataset_label,
            dataset=dataset,
            filter_rate=filter_rate,
            category_nums=category_nums,
            category_column_idxes=category_column_idxes,
            name=name,
            description=description,
            **kwargs,
        )

    def get_ground_truth_file(self) -> str:
        category_nums = self.category_nums
        if len(category_nums) == 1:
            file_name = f"neighbors_{category_num_to_column_name(category_nums[0]).replace('-', '_')}_filter.parquet"
        else:
            file_name = f"neighbors_{self.name}.parquet"
        return file_name


class CustomOrFilter(CustomFilter):
    case_id: CaseType = CaseType.CustomOrFilter
    filter_rate: float
    category_nums: list[int]
    category_column_idxes: list[int]

    def __init__(
        self,
        dataset_label: ScalarDatasetLabel | str,
        category_nums: list[int],
        **kwargs,
    ):
        r = 0.0
        for category_num in category_nums:
            r += 1.0 / category_num
        filter_rate = 1.0 - r
        dataset = ScalarDatasetLabel(dataset_label).get_dataset()
        category_column_idxes = [
            category_num_to_category_column_idx(dataset, category_num)
            for category_num in category_nums
        ]
        name = f"{len(category_nums)}_or_{'_'.join([f'{category_num}' for category_num in category_nums])}"
        description = f"{dataset.data.name} - {name}"
        super().__init__(
            dataset_label=dataset_label,
            dataset=dataset,
            filter_rate=filter_rate,
            category_nums=category_nums,
            category_column_idxes=category_column_idxes,
            name=name,
            description=description,
            **kwargs,
        )

    def get_ground_truth_file(self) -> str:
        category_nums = self.category_nums
        if len(category_nums) == 1:
            file_name = f"neighbors_{category_num_to_column_name(category_nums[0]).replace('-', '_')}_filter.parquet"
        else:
            category_nums.sort()
            file_name = f"neighbors_{self.name}.parquet"
        return file_name


type2case = {
    CaseType.CapacityDim960: CapacityDim960,
    CaseType.CapacityDim128: CapacityDim128,
    CaseType.Performance768D100M: Performance768D100M,
    CaseType.Performance768D10M: Performance768D10M,
    CaseType.Performance768D1M: Performance768D1M,
    CaseType.Performance768D10M1P: Performance768D10M1P,
    CaseType.Performance768D1M1P: Performance768D1M1P,
    CaseType.Performance768D10M99P: Performance768D10M99P,
    CaseType.Performance768D1M99P: Performance768D1M99P,
    CaseType.Performance1536D500K: Performance1536D500K,
    CaseType.Performance1536D5M: Performance1536D5M,
    CaseType.Performance1536D500K1P: Performance1536D500K1P,
    CaseType.Performance1536D5M1P: Performance1536D5M1P,
    CaseType.Performance1536D500K99P: Performance1536D500K99P,
    CaseType.Performance1536D5M99P: Performance1536D5M99P,
    CaseType.PerformanceGlove200: PerformanceGlove200,
    CaseType.PerformanceLastFM: PerformanceLastFM,
    CaseType.PerformanceGIST768Internal: PerformanceGIST768Internal,
    CaseType.PerformanceText2imgInternal: PerformanceText2imgInternal,
    CaseType.PerformanceCohereInternal: PerformanceCohereInternal,
    CaseType.PerformanceOpenAIInternal: PerformanceOpenAIInternal,
    CaseType.CustomIntFilter: CustomIntFilter,
    CaseType.CustomCategoryFilter: CustomCategoryFilter,
    CaseType.CustomAndFilter: CustomAndFilter,
    CaseType.CustomOrFilter: CustomOrFilter,
}
