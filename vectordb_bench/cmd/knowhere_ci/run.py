from vectordb_bench.cmd.knowhere_ci.config import drop_old
from vectordb_bench.cmd.knowhere_ci.knowhere_tasks import get_tasks
from vectordb_bench.cmd.knowhere_ci.results import get_results
from vectordb_bench.interface import benchMarkRunner


def run_for_knowhere(with_cardinal: bool = False):
    task_label = "cardinal_ci" if with_cardinal else "knowhere_ci"
    tasks = get_tasks(with_cardinal)
    print("get tasks", tasks)
    benchMarkRunner.set_drop_old(drop_old)
    benchMarkRunner.run(tasks, task_label)


def get_knowhere_test_results():
    results = get_results(with_cardinal=False)
    return results


def get_cardinal_test_results():
    results = get_results(with_cardinal=True)
    return results
