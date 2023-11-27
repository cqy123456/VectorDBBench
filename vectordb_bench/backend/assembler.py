from vectordb_bench.backend.clients.api import TestType
from .cases import CaseLabel
from .task_runner import CaseRunner, RunningStatus, TaskRunner
from ..models import TaskConfig
from ..backend.clients import EmptyDBCaseConfig
import logging


log = logging.getLogger(__name__)


class Assembler:
    @classmethod
    def assemble(cls, run_id, task: TaskConfig) -> CaseRunner:
        c_cls = task.case_config.case_id.case_cls

        c = c_cls(task.case_config.custom_case)
        if type(task.db_case_config) != EmptyDBCaseConfig:
            task.db_case_config.metric_type = c.dataset.data.metric_type

        if task.db_config.test_type == TestType.LIBRARY and c.dataset.data.use_shuffled:
            log.warning(f"Should not use shuffle data for library tests. Force set use_shuffled to False")
            c.dataset.data.use_shuffled = False

        groundtruth_file = c.get_ground_truth_file()
        if c.dataset.data.check_s3 or c.dataset.check_case_has_groundtruth(groundtruth_file):
            runner = CaseRunner(
                run_id=run_id,
                config=task,
                ca=c,
                status=RunningStatus.PENDING,
            )
            return runner
        else:
            log.info(f"Skip Case - No groundtruth file {groundtruth_file}")
            return None

    @classmethod
    def assemble_all(
        cls, run_id: str, task_label: str, tasks: list[TaskConfig]
    ) -> TaskRunner:
        """group by case type, db, and case dataset"""
        runners = [cls.assemble(run_id, task) for task in tasks]
        runners = [runner for runner in runners if runner is not None]
        load_runners = [r for r in runners if r.ca.label == CaseLabel.Load]
        perf_runners = [r for r in runners if r.ca.label == CaseLabel.Performance]

        # group by db
        db2runner = {}
        for r in perf_runners:
            db = r.config.db
            if db not in db2runner:
                db2runner[db] = []
            db2runner[db].append(r)

        # check dbclient installed
        for k in db2runner.keys():
            _ = k.init_cls

        # sort by dataset size
        for k in db2runner.keys():
            db2runner[k].sort(key=lambda x: x.ca.dataset.data.size)

        all_runners = []
        all_runners.extend(load_runners)
        for v in db2runner.values():
            all_runners.extend(v)

        return TaskRunner(
            run_id=run_id,
            task_label=task_label,
            case_runners=all_runners,
        )
