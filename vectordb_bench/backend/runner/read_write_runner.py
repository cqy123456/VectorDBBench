import logging
from typing import Iterable
import multiprocessing as mp
import concurrent
import numpy as np
import math

from .mp_runner import MultiProcessingSearchRunner
from .serial_runner import SerialSearchRunner
from .rate_runner import RatedMultiThreadingInsertRunner
from vectordb_bench.backend.clients import api
from vectordb_bench.backend.dataset import DatasetManager

log = logging.getLogger(__name__)


class ReadWriteRunner(MultiProcessingSearchRunner, RatedMultiThreadingInsertRunner):
    def __init__(
        self,
        db: api.VectorDB,
        dataset: DatasetManager,
        insert_rate: int = 1000,
        normalize: bool = False,
        k: int = 100,
        filters: dict | None = None,
        concurrencies: Iterable[int] = (1, 15, 50),
        search_stage: Iterable[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        stage_search_dur: int = 120, # seconds, TODO, choose duration based on dur of insert 10% data
        flush_percent: float = 1.0,
        timeout: float | None = None,
    ):
        self.insert_rate = insert_rate
        self.data_volume = dataset.data.size
        self.search_stage = search_stage
        self.state_search_dur = stage_search_dur

        log.info(f"Init runner, concurencys={concurrencies}, search_stage={search_stage}, stage_search_dur={stage_search_dur}")

        test_emb = np.stack(dataset.test_data["emb"])
        if normalize:
            test_emb = test_emb / np.linalg.norm(test_emb, axis=1)[:, np.newaxis]
        test_emb = test_emb.tolist()

        self.total_batch = math.ceil(self.data_volume / self.insert_rate)

        MultiProcessingSearchRunner.__init__(
            self,
            db=db,
            test_data=test_emb,
            k=k,
            filters=filters,
            concurrencies=concurrencies,
            duration=stage_search_dur,
        )
        RatedMultiThreadingInsertRunner.__init__(
            self,
            rate=insert_rate,
            db=db,
            dataset_iter=iter(dataset),
            batch_num=self.total_batch,
            normalize=normalize,
            flush_percent=flush_percent,
        )
        self.serial_search_runner = SerialSearchRunner(
            db=db,
            test_data=test_emb,
            ground_truth=dataset.gt_data,
            k=k,
        )

    def run_read_write(self):
        futures = []
        with mp.Manager() as m:
            q = m.Queue()
            flush_sig = m.Queue()
            with concurrent.futures.ProcessPoolExecutor(mp_context=mp.get_context("spawn"), max_workers=2) as executor:
                futures.append(executor.submit(self.run_with_rate, q))
                futures.append(executor.submit(self.run_search_by_sig, q, flush_sig))

                for future in concurrent.futures.as_completed(futures):
                    res = future.result()
                    log.info(f"Result = {res}")

        log.info("Concurrent read write all done")

    def run_search_by_sig(self, q, flush_sig):
        res = []
        batch = 1
        recall = 'x'

        for stage in self.search_stage:
            target_batch = int(self.total_batch * stage)
            while q.get(block=True):
                if batch >= target_batch:
                    perc = int(stage * 100)
                    log.info(f"Insert {perc}% done, total batch={self.total_batch}")
                    log.info(f"Serial search - {perc}% start")
                    recall, ndcg, p99 =self.serial_search_runner.run()

                    log.info(f"Conc search - {perc}% start")
                    max_qps = self.run_by_dur(self.duration)
                    res.append((perc, max_qps, recall))
                    break

                batch += 1

        log.info("Insert 100% done, start optimize")
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._task)
            _ = future.result()

        log.info(f"Optimize done, {res}, start conc search")
        max_qps = self.run_by_dur(300)
        res.append((1000, max_qps, recall))

        return res

    def _task(self) -> None:
        with self.db.init():
            self.db.optimize()
