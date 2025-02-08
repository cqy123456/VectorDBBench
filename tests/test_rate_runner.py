from typing import Iterable
import argparse
from vectordb_bench.backend.dataset import Dataset, DatasetSource
from vectordb_bench.backend.runner.rate_runner import RatedMultiThreadingInsertRunner
from vectordb_bench.backend.runner.read_write_runner import ReadWriteRunner
from vectordb_bench.backend.clients import DB, VectorDB
from vectordb_bench.backend.clients.milvus.config import HNSWConfig
from vectordb_bench.backend.clients.zilliz_cloud.config import AutoIndexConfig

import logging

log = logging.getLogger("vectordb_bench")
log.setLevel(logging.DEBUG)

def get_rate_runner(db):
    cohere = Dataset.COHERE.manager(100_000)
    prepared = cohere.prepare(DatasetSource.AliyunOSS)
    assert prepared
    runner = RatedMultiThreadingInsertRunner(
        rate = 10,
        db = db,
        dataset = cohere,
    )

    return runner

def test_rate_runner(db, insert_rate):
    runner = get_rate_runner(db)

    _, t = runner.run_with_rate()
    log.info(f"insert run done, time={t}")

def test_read_write_runner(db, insert_rate, conc: list, search_stage: Iterable[float], stage_search_dur: int, local: bool=False, flush_percent: float=1.0):
    cohere = Dataset.COHERE.manager(10_000_000)
    if local is True:
        source = DatasetSource.AliyunOSS
    else:
        source = DatasetSource.S3
    prepared = cohere.prepare(source)
    assert prepared

    rw_runner = ReadWriteRunner(
        db=db,
        dataset=cohere,
        insert_rate=insert_rate,
        search_stage=search_stage,
        stage_search_dur=stage_search_dur,
        concurrencies=conc,
        flush_percent=flush_percent
    )
    rw_runner.run_read_write()


def get_db(db: str, config: dict) -> VectorDB:
    if db == DB.Milvus.name:
        return DB.Milvus.init_cls(dim=768, db_config=config, db_case_config=HNSWConfig(metric_type="L2", M=36, efConstruction=128,ef=110), drop_old=True, pre_load=True)
    elif db == DB.ZillizCloud.name:
        return DB.ZillizCloud.init_cls(dim=768, db_config=config, db_case_config=AutoIndexConfig(metric_type="L2"), drop_old=True, pre_load=True)
    else:
        raise ValueError(f"unknown db: {db}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--insert_rate", type=int, default="1000", help="insert entity row count per seconds, cps")
    parser.add_argument("-d", "--db", type=str, default=DB.Milvus.name, help="db name")
    #  parser.add_argument("-c", "--conc", type=list, default=(1, 15, 50), help="search conc")
    #  parser.add_argument("-s", "--search_stage", type=list, default=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0), help="search stage")
    parser.add_argument("-t", "--duration", type=int, default=120, help="stage search duration in seconds")
    parser.add_argument("--use_s3", action='store_true', help="whether to use S3 dataset")
    parser.add_argument("-p", "--flush_percent", type=float, default=1.0, help="Insert percentage to flush")

    flags = parser.parse_args()

    # TODO read uri, user, password from .env
#     config = {
#         # debug
#         #"uri":"https://in01-615961829bf1848.gcp-us-west1.vectordb-uat3.zillizcloud.com:443",
#         #"user": "root",
#         #"password": "p8.qhT+0=GGv%[@-Pxp!OJGn/halvmQd",
#         # release 
#         #"uri":"https://in01-81de0b2fec177b4.gcp-us-west1.vectordb-uat3.zillizcloud.com",
#         #"user":"root",
#         #"password":"M0-dTfn26Y:q?b[@&1EM=24vcsK7e}c[",
#         "uri":"https://in01-0a9943bfff1f434.gcp-us-west1.vectordb-uat3.zillizcloud.com:443",
#         "user":"root",
#         "password":"C4%}}PBuk0,A&bdBUm06]=x|s/g$S}Dz",
#    }
    config = {
         "uri":"http://localhost:19530",
    }

    #  conc = (1, 15, 50)
    #  search_stage = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

    conc = (50,)
    search_stage = (0.0,)

    db = get_db(flags.db, config)
    test_read_write_runner(
        db=db,
        insert_rate=flags.insert_rate,
        conc=conc,
        search_stage=search_stage,
        stage_search_dur=flags.duration,
        local=flags.use_s3,
        flush_percent=flags.flush_percent,
    )

