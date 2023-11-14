from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.dataset import ScalarDatasetLabel
from vectordb_bench.base import BaseModel
from vectordb_bench.frontend.utils import get_all_combinations
from vectordb_bench.models import CaseConfig


class BatchCasesOption(BaseModel):
    label: str
    description: str
    cases: list[CaseConfig]


def get_all_int_filter_cases(dataset_label: ScalarDatasetLabel):
    dataset = dataset_label.get_dataset().data
    return BatchCasesOption(
        label=f"[Batch Int-Filter] {dataset.name} ({dataset.metric_type.value}) {dataset.dim} dim * {dataset.size:,}",
        description=f"filter_rates: [None, 0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99]",
        cases = [
            CaseConfig(
                case_id=CaseType.CustomIntFilter,
                custom_case=dict(dataset_label=dataset_label, filter_rate=int_filter_rate)
            )
            for int_filter_rate in dataset.int_filter_rates
        ]
    )

def get_all_category_filter_cases(dataset_label: ScalarDatasetLabel):
    dataset = dataset_label.get_dataset().data
    return BatchCasesOption(
        label=f"[Batch Category-Filter] {dataset.name} ({dataset.metric_type.value}) {dataset.dim} dim * {dataset.size:,}",
        description=f"""category_nums: [2, 5, 10, 100, 1000]; filter_rates: [0.5, 0.2, 0.1, 0.01, 0.001];""",
        cases = [
            CaseConfig(
                case_id=CaseType.CustomCategoryFilter,
                custom_case=dict(dataset_label=dataset_label, category_num=category_num)
            )
            for category_num in dataset.category_nums
        ]
    )

# and- / or- cases 
def get_all_complex_cases(caseType: CaseType, dataset_label: ScalarDatasetLabel, label=""):
    dataset = dataset_label.get_dataset().data
    return BatchCasesOption(
        label=f"[Batch {label}-Filter] {dataset.name} ({dataset.metric_type.value}) {dataset.dim} dim * {dataset.size:,}",
        description=f"""Randomly pick different category-columns to combine (All-{label}); category_nums: [2, 5, 10, 100, 1000];""",
        cases = [
            CaseConfig(
                case_id=caseType,
                custom_case=dict(dataset_label=dataset_label, category_nums=category_nums)
            )
            for category_nums in get_all_combinations(dataset.category_nums)
        ]
    )

def get_all_and_filter_cases(dataset_label: ScalarDatasetLabel):
    return get_all_complex_cases(CaseType.CustomAndFilter, dataset_label, 'And')

def get_all_or_filter_cases(dataset_label: ScalarDatasetLabel):
    return get_all_complex_cases(CaseType.CustomOrFilter, dataset_label, 'Or')