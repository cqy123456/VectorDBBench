from vectordb_bench.interface import benchMarkRunner
from vectordb_bench.cmd.knowhere_ci.config import task_label
from vectordb_bench.models import CaseResult
import json


def get_results():
    _results = benchMarkRunner.get_results()
    results = [res for res in _results if res.task_label == task_label]
    case_results: list[CaseResult] = []
    for res in results:
        _case_results = res.results
        case_results += _case_results

    dataList = []
    for case_result in case_results:
        data = {}

        metrics = case_result.metrics
        task_config = case_result.task_config

        data["search_vps"] = metrics.qps
        data["search_recall"] = metrics.recall
        data["build_time"] = metrics.load_duration

        case_config = task_config.case_config
        case = case_config.case_id.case_cls(case_config.custom_case)
        dataset = case.dataset.data
        data["dataset_name"] = dataset.name
        data["data_rows"] = dataset.size
        data["data_dim"] = dataset.dim
        data["metric_type"] = dataset.metric_type.value

        db_config = task_config.db_config
        data["index_type"] = db_config.index_type
        build_params = db_config.config_json
        del build_params["index_type"]
        data["build_params"] = json.dumps(build_params)

        db_case_config = task_config.db_case_config
        search_params = db_case_config.search_param()
        del search_params["metric_type"]
        data["search_params"] = json.dumps(search_params)

        dataList.append(data)

    return dataList
