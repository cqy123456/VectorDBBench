from vectordb_bench.cmd.knowhere_ci.config import task_label, drop_old
from vectordb_bench.cmd.knowhere_ci.knowhere_tasks import get_tasks
from vectordb_bench.cmd.knowhere_ci.results import get_results
from vectordb_bench.interface import benchMarkRunner


def run_for_knowhere():
    tasks = get_tasks()
    benchMarkRunner.set_drop_old(drop_old)
    benchMarkRunner.run(tasks, task_label)


def get_knowhere_test_results():
    results = get_results()
    return results
