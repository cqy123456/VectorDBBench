import logging
import psutil
import traceback
import concurrent
import numpy as np
from enum import Enum, auto

from vectordb_bench.backend.dataset import category_num_to_column_name

from . import utils
from .cases import Case, CaseLabel, CaseType
from ..base import BaseModel
from ..models import TaskConfig, PerformanceTimeoutError

from .clients import api, MetricType
from ..metric import Metric
from .runner import MultiProcessingSearchRunner
from .runner import SerialSearchRunner, SerialInsertRunner


log = logging.getLogger(__name__)


class RunningStatus(Enum):
    PENDING = auto()
    FINISHED = auto()


class CaseRunner(BaseModel):
    """DataSet, filter_rate, db_class with db config

    Fields:
        run_id(str): run_id of this case runner,
            indicating which task does this case belong to.
        config(TaskConfig): task configs of this case runner.
        ca(Case): case for this case runner.
        status(RunningStatus): RunningStatus of this case runner.

        db(api.VectorDB): The vector database for this case runner.
    """

    run_id: str
    config: TaskConfig
    ca: Case
    status: RunningStatus

    db: api.VectorDB | None = None
    test_emb: list[list[float]] | None = None
    search_runner: MultiProcessingSearchRunner | None = None
    serial_search_runner: SerialSearchRunner | None = None

    def __eq__(self, obj):
        if isinstance(obj, CaseRunner):
            return (
                self.ca.label == CaseLabel.Performance
                and self.ca.with_category_column == obj.ca.with_category_column
                and self.config.db == obj.config.db
                and self.config.db_case_config == obj.config.db_case_config
                and self.ca.dataset == obj.ca.dataset
            )
        return False

    def display(self) -> dict:
        c_dict = self.ca.dict(
            include={"label": True, "filters": True, "dataset": {"data": True}}
        )
        c_dict["db"] = self.config.db_name
        return c_dict

    @property
    def normalize(self) -> bool:
        assert self.db
        return (
            self.db.need_normalize_cosine()
            and self.ca.dataset.data.metric_type == MetricType.COSINE
        )

    def init_db(self, drop_old: bool = True) -> None:
        db_cls = self.config.db.init_cls

        dataset = self.ca.dataset.data
        # some dataset have category_scalars, but case doesn't need.
        category_column_names = dataset.category_column_names
        with_category_column = self.ca.with_category_column
        if with_category_column and len(category_column_names) == 0:
            raise RuntimeError(f"No category_columns for the case - {self.ca.name}")
        int_column_names = dataset.int_column_names

        self.db = db_cls(
            dim=dataset.dim,
            db_config=self.config.db_config.to_dict(),
            db_case_config=self.config.db_case_config,
            drop_old=drop_old,
            with_category_column=with_category_column,
            category_column_names=category_column_names,
            int_column_names=int_column_names,
        )

    def _pre_run(self, drop_old: bool = True):
        try:
            self.init_db(drop_old)
            self.ca.dataset.prepare()
        except ModuleNotFoundError as e:
            log.warning(
                f"pre run case error: please install client for db: {self.config.db}, error={e}"
            )
            raise e from None
        except Exception as e:
            log.warning(f"pre run case error: {e}")
            raise e from None

    def run(self, drop_old: bool = True) -> Metric:
        self._pre_run(drop_old)

        if self.ca.label == CaseLabel.Load:
            return self._run_capacity_case()
        elif self.ca.label == CaseLabel.Performance:
            return self._run_perf_case(drop_old)
        else:
            msg = f"unknown case type: {self.ca.label}"
            log.warning(msg)
            raise ValueError(msg)

    def _run_capacity_case(self) -> Metric:
        """run capacity cases

        Returns:
            Metric: the max load count
        """
        test_type = self.config.db_config.test_type
        log.info(f"Test Type: {test_type}")
        if test_type == api.TestType.LIBRARY:
            msg = "Capacity test only support Database, not Library."
            log.warning(msg)
            raise ValueError(msg)

        log.info("Start capacity case")
        try:
            runner = SerialInsertRunner(
                self.db, self.ca.dataset, self.normalize, self.ca.load_timeout
            )
            count = runner.run_endlessness()
        except Exception as e:
            log.warning(f"Failed to run capacity case, reason = {e}")
            raise e from None
        else:
            log.info(
                f"Capacity case loading dataset reaches VectorDB's limit: max capacity = {count}"
            )
            return Metric(max_load_count=count)

    def _run_perf_case(self, drop_old: bool = True) -> Metric:
        """run performance cases

        Returns:
            Metric: load_duration, recall, serial_latency_p99, and, qps
        """
        try:
            m = Metric()
            if drop_old:
                _, load_dur = self._load_train_data()
                build_dur = self._optimize()
                m.load_duration = round(load_dur + build_dur, 4)
                log.info(
                    f"Finish loading the entire dataset into VectorDB,"
                    f" insert_duration={load_dur}, optimize_duration={build_dur}"
                    f" load_duration(insert + optimize) = {m.load_duration}"
                )

            self._init_search_runner()
            test_type = self.config.db_config.test_type
            serial_search_results = self._serial_search(test_type)
            if test_type == api.TestType.DATABASE:
                m.recall, m.serial_latency_p99 = serial_search_results
                m.qps = self._conc_search()
            if test_type == api.TestType.LIBRARY:
                m.recall, m.qps = serial_search_results
        except Exception as e:
            log.warning(f"Failed to run performance case, reason = {e}")
            traceback.print_exc()
            raise e from None
        else:
            log.info(f"Performance case got result: {m}")
            return m

    @utils.time_it
    def _load_train_data(self):
        """Insert train data and get the insert_duration"""
        try:
            test_type = self.config.db_config.test_type
            runner = SerialInsertRunner(
                self.db, self.ca.dataset, self.normalize, self.ca.load_timeout
            )
            runner.run(test_type)
        except Exception as e:
            raise e from None
        finally:
            runner = None

    def _serial_search(
        self, test_type: api.TestType = api.TestType.DATABASE
    ) -> tuple[float, float]:
        """Performance serial tests, search the entire test data once,
        calculate the recall, serial_latency_p99

        Returns:
            tuple[float, float]: recall, serial_latency_p99
        """
        try:
            return self.serial_search_runner.run(test_type)
        except Exception as e:
            log.warning(f"search error: {str(e)}, {e}")
            self.stop()
            raise e from None

    def _conc_search(self):
        """Performance concurrency tests, search the test data endlessness
        for 30s in several concurrencies

        Returns:
            float: the largest qps in all concurrencies
        """
        try:
            return self.search_runner.run()
        except Exception as e:
            log.warning(f"search error: {str(e)}, {e}")
            raise e from None
        finally:
            self.stop()

    @utils.time_it
    def _task(self) -> None:
        log.info("Optimize start")
        with self.db.init():
            self.db.optimize()
        log.info("Optimize finished")

    def _optimize(self) -> float:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._task)
            try:
                return future.result(timeout=self.ca.optimize_timeout)[1]
            except TimeoutError as e:
                log.warning(f"VectorDB optimize timeout in {self.ca.optimize_timeout}")
                for pid, _ in executor._processes.items():
                    psutil.Process(pid).kill()
                raise PerformanceTimeoutError(
                    "Performance case optimize timeout"
                ) from e
            except Exception as e:
                log.warning(f"VectorDB optimize error: {e}")
                raise e from None

    def _init_search_runner(self):
        test_emb = np.stack(self.ca.dataset.test_data["emb"])
        if self.normalize:
            test_emb = test_emb / np.linalg.norm(test_emb, axis=1)[:, np.newaxis]
        self.test_emb = test_emb.tolist()

        gt_df = self.ca.dataset.get_ground_truth(self.ca.get_ground_truth_file())

        filters = self.db.get_filters(self.ca)

        test_type = self.config.db_config.test_type

        # filter_bitset should be calculated in advance when test Library
        # - reload train data to get the vaild ids according to filter-expr
        # - convert the valid ids to bitset that the client can recognize
        valid_ids = None
        if (
            self.ca.filter_rate is not None
            and self.ca.filter_rate > 0.0
            and test_type == api.TestType.LIBRARY
        ):
            valid_ids = self._get_valid_ids()

        self.serial_search_runner = SerialSearchRunner(
            db=self.db,
            test_data=self.test_emb,
            ground_truth=gt_df,
            filters=filters,
            valid_ids=valid_ids,
        )

        self.search_runner = MultiProcessingSearchRunner(
            db=self.db,
            test_data=self.test_emb,
            filters=filters,
        )

    def _get_valid_ids(self) -> list[int]:
        """return checked valid id"""
        case = self.ca
        log.info(
            f"According the filter-expr {case.name}, reload the train data to get valid ids."
        )
        import polars as pl

        file = self.ca.dataset.data_dir / "only_id_and_scalars.parquet"
        df = pl.read_parquet(file)
        # pl_filter = pl.col("id") >= self.ca.filters.get("id")
        pl_filter = None
        if case.case_id == CaseType.CustomCategoryFilter:
            pl_filter = pl.col(category_num_to_column_name(case.category_num)) == 1
        elif case.case_id == CaseType.CustomAndFilter:
            pl_filter = pl.col(category_num_to_column_name(case.category_nums[0])) == 1
            for i in range(1, len(case.category_nums)):
                category_num = case.category_nums[i]
                pl_filter = pl_filter & (
                    pl.col(category_num_to_column_name(category_num)) == 1
                )
        elif case.case_id == CaseType.CustomOrFilter:
            pl_filter = pl.col(category_num_to_column_name(case.category_nums[0])) == 1
            for i in range(1, len(case.category_nums)):
                category_num = case.category_nums[i]
                pl_filter = pl_filter | (
                    pl.col(category_num_to_column_name(category_num)) == 1
                )
        else:
            pl_filter = (
                pl.col(case.dataset.data.int_column_names[0])
                >= case.dataset.data.size * case.filter_rate
            )

        filter_df = df.filter(pl_filter)
        del df
        valid_ids = filter_df["id"].to_list()
        log.info("Get valid ids successfully.")
        return valid_ids

    def stop(self):
        if self.search_runner:
            self.search_runner.stop()


class TaskRunner(BaseModel):
    run_id: str
    task_label: str
    case_runners: list[CaseRunner]

    def num_cases(self) -> int:
        return len(self.case_runners)

    def num_finished(self) -> int:
        return self._get_num_by_status(RunningStatus.FINISHED)

    def set_finished(self, idx: int) -> None:
        self.case_runners[idx].status = RunningStatus.FINISHED

    def _get_num_by_status(self, status: RunningStatus) -> int:
        return sum([1 for c in self.case_runners if c.status == status])

    def display(self) -> None:
        DATA_FORMAT = " %-14s | %-12s %-20s %7s | %-10s"
        TITLE_FORMAT = (" %-14s | %-12s %-20s %7s | %-10s") % (
            "DB",
            "CaseType",
            "Dataset",
            "CaseName",
            "task_label",
        )

        fmt = [TITLE_FORMAT]
        fmt.append(DATA_FORMAT % ("-" * 11, "-" * 12, "-" * 20, "-" * 7, "-" * 7))

        for f in self.case_runners:
            caseName = f.ca.name
            ds_str = f"{f.ca.dataset.data.name}-{f.ca.dataset.data.label}-{utils.numerize(f.ca.dataset.data.size)}"
            fmt.append(
                DATA_FORMAT
                % (
                    f.config.db_name,
                    f.ca.label.name,
                    ds_str,
                    caseName,
                    self.task_label,
                )
            )

        tmp_logger = logging.getLogger("no_color")
        for f in fmt:
            tmp_logger.info(f)
