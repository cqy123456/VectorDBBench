from dataclasses import asdict
from vectordb_bench.backend.cases import CaseType
from vectordb_bench.interface import benchMarkRunner
from vectordb_bench.models import CaseResult


def getChartsData():
    allResults = benchMarkRunner.get_results()
    singleFilterCaseResults: list[CaseResult] = []

    for res in allResults:
        results = res.results
        for result in results:
            case_config = result.task_config.case_config
            if case_config.case_id in [CaseType.CustomAndFilter, CaseType.CustomOrFilter]:
                singleFilterCaseResults.append(result)
                    

    singleFilterData = formatData(singleFilterCaseResults)
    return singleFilterData


def formatData(caseResults: list[CaseResult]):
    data = []
    for caseResult in caseResults:
        db = caseResult.task_config.db.value
        dbLabel = caseResult.task_config.db_config.db_label
        dbName = caseResult.task_config.db_name
        case_config = caseResult.task_config.case_config
        case = case_config.case_id.case_cls(case_config.custom_case)
        filter_rate = case.filter_rate
        filter_type = "and" if case.case_id == CaseType.CustomAndFilter else "or"
        dataset = case.dataset.data.name
        clause_num = len(case.category_nums)
        filter_fields = case.category_nums
        metrics = asdict(caseResult.metrics)
        data.append(
            {
                "db": db,
                "dbLabel": dbLabel,
                "dbName": dbName,
                "dataset": dataset,
                "filter_type": filter_type,
                "filter_rate": filter_rate if filter_rate is not None else -0.1,
                "clause_num": f"{clause_num}",
                "type_and_clause_num": f"{filter_type}-{clause_num}",
                "dbLabel_and_clause_num": f"{dbLabel}-{clause_num}",
                "filter_fields": filter_fields,
                **metrics,
            }
        )
    return data
