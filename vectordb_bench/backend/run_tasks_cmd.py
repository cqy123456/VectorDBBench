from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.milvus.config import HNSWConfig, MilvusConfig
from vectordb_bench.interface import benchMarkRunner
import logging
from vectordb_bench import config
from vectordb_bench.models import CaseConfig, TaskConfig

log = logging.getLogger("vectordb_bench")


def main():
    tasks: list[TaskConfig] = [
        TaskConfig(
            db=DB.Milvus,
            db_config=MilvusConfig(
                uri="http://10.104.22.209:19530",
                db_label="",
            ),
            case_config=CaseConfig(case_id=CaseType.Performance768D1M),
            db_case_config=HNSWConfig(
                M=30,
                efConstruction=360,
                ef=100,
            ),
        )
    ]
    drop_old = True
    taskLabel = "test_cmd"
    benchMarkRunner.set_drop_old(drop_old)
    benchMarkRunner.run(tasks, taskLabel)


if __name__ == "__main__":
    main()
