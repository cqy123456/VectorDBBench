
from vectordb_bench.backend.clients.api import IndexType
from vectordb_bench.models import CaseConfig, CaseConfigParamType, TaskConfig


def generate_tasks(activedDbList, dbConfigs, activedCaseList: list[CaseConfig], allCaseConfigs):
    tasks = []
    for db in activedDbList:
        for case in activedCaseList:
            task = TaskConfig(
                db=db.value,
                db_config=dbConfigs[db],
                case_config=case,
                db_case_config=db.case_config_cls(
                    allCaseConfigs[db][case.case_id].get(
                        CaseConfigParamType.IndexType, IndexType.HNSW
                    )
                )(
                    caseConfig=case,
                    **{
                        key.value: value
                        for key, value in allCaseConfigs[db][case.case_id].items()
                    }
                ),
            )
            tasks.append(task)

    return tasks
