from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.knowhere.config import (
    KnowhereConfig,
    KnowhereIndexConfig,
)
from vectordb_bench.models import CaseConfig, TaskConfig
import json
from vectordb_bench.cmd.knowhere_ci.config import (
    case_ids,
    build_threads,
    search_threads,
    get_ivfflat_params,
    get_hnsw_params,
    get_diskann_params,
)


def get_tasks(with_cardinal: bool = False) -> list[TaskConfig]:
    tasks: list[TaskConfig] = []

    tasks += knowhere_ivfflat_tasks(case_ids=case_ids,
                                    with_cardinal=with_cardinal)
    tasks += knowhere_hnsw_tasks(case_ids=case_ids,
                                 with_cardinal=with_cardinal)
    tasks += knowhere_diskann_tasks(case_ids=case_ids,
                                    with_cardinal=with_cardinal)

    return tasks


def knowhere_ivfflat_tasks(case_ids: list[CaseType], with_cardinal: bool = False) -> list[TaskConfig]:
    if with_cardinal:
        return []
    tasks: list[TaskConfig] = []
    db = DB.Knowhere
    index_type = "IVFFLAT"
    ivfflat_params = get_ivfflat_params()
    build_config = ivfflat_params["build"]
    db_label = f"cardinal-{index_type}" if with_cardinal else f"knowhere-{index_type}"
    db_config = KnowhereConfig(
        index_type=index_type,
        config=json.dumps(build_config)[1:-1],
        db_label=db_label,
        build_threads=build_threads,
        search_threads=search_threads,
    )

    for case_id in case_ids:
        for nprobe in ivfflat_params["search"]["nprobe"]:
            db_case_config = KnowhereIndexConfig(nprobe=nprobe)
            tasks.append(
                TaskConfig(
                    db=db,
                    db_config=db_config,
                    case_config=CaseConfig(case_id=case_id, custom_case={}),
                    db_case_config=db_case_config,
                )
            )

    return tasks


def knowhere_hnsw_tasks(case_ids: list[CaseType], with_cardinal: bool = False) -> list[TaskConfig]:
    tasks: list[TaskConfig] = []
    db = DB.Knowhere
    index_type = "HNSW"
    hnsw_params = get_hnsw_params(with_cardinal=with_cardinal)
    build_config = hnsw_params["build"]
    db_label = f"cardinal-{index_type}" if with_cardinal else f"knowhere-{index_type}"
    db_config = KnowhereConfig(
        index_type=index_type,
        config=json.dumps(build_config)[1:-1],
        db_label=db_label,
        build_threads=build_threads,
        search_threads=search_threads,
    )

    for case_id in case_ids:
        for ef in hnsw_params["search"]["ef"]:
            db_case_config = KnowhereIndexConfig(ef=ef)
            tasks.append(
                TaskConfig(
                    db=db,
                    db_config=db_config,
                    case_config=CaseConfig(case_id=case_id, custom_case={}),
                    db_case_config=db_case_config,
                )
            )

    return tasks


def knowhere_diskann_tasks(case_ids: list[CaseType], with_cardinal: bool = False) -> list[TaskConfig]:
    """use knowhere client to test knowherw-diskann"""
    tasks: list[TaskConfig] = []
    db = DB.Knowhere
    index_type = "DISKANN"

    for case_id in case_ids:
        case = case_id.case_cls()
        dataset = case.dataset.data
        dim = dataset.dim
        num_rows = dataset.size
        diskann_params = get_diskann_params(
            num_rows=num_rows, dim=dim, with_cardinal=with_cardinal)
        build_config = diskann_params["build"]
        db_label = f"cardinal-{index_type}" if with_cardinal else f"knowhere-{index_type}"
        db_config = KnowhereConfig(
            index_type=index_type,
            config=json.dumps(build_config)[1:-1],
            db_label=db_label,
            build_threads=build_threads,
            search_threads=search_threads,
        )

        for search_list_size in diskann_params["search"]["search_list_size"]:
            db_case_config = KnowhereIndexConfig(
                search_list_size=search_list_size)
            tasks.append(
                TaskConfig(
                    db=db,
                    db_config=db_config,
                    case_config=CaseConfig(case_id=case_id, custom_case={}),
                    db_case_config=db_case_config,
                )
            )

    return tasks
