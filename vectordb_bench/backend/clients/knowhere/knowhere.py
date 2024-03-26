import logging
from contextlib import contextmanager
from typing import Type
from ..api import VectorDB, DBCaseConfig, DBConfig, IndexType
from .config import KnowhereConfig, KnowhereIndexConfig
import pathlib
import json
import os
import struct

log = logging.getLogger(__name__)


class Knowhere(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: KnowhereIndexConfig,
        drop_old: bool = False,
        name: str = "Knowhere",
        vectors_file="train_vectors.fbin",
        tmp_dir_path="./vectordb_bench/results/tmp_knowhere/",
        **kwargs,
    ):
        self.name = name
        self.db_config = db_config
        self.case_config = db_case_config
        self.dim = dim
        self.config = json.loads(f'{{{self.db_config.get("config")}}}')
        self.index_type = self.db_config.get("index_type")
        self.config["dim"] = dim
        self.build_threads = db_config.get("build_threads", 2)
        self.search_threads = db_config.get("search_threads", 2)

        import knowhere
        knowhere.SetBuildThreadPool(self.build_threads)
        knowhere.SetSearchThreadPool(self.search_threads)

        self.version = knowhere.GetCurrentVersion()
        log.info(f"knowhere version - {self.version}")
        self.index_file_name = (
            self.index_type
            + "_"
            + db_config.get("config")
            .replace('"', "")
            .replace(" ", "")
            .replace(":", "_")
            .replace(",", "_")
        )
        self.index_dir = pathlib.Path(tmp_dir_path)
        if not self.index_dir.exists():
            self.index_dir.mkdir(parents=True)

        self.index_file_path = (
            self.index_dir / self.index_file_name).as_posix()
        self.vectors_file = (self.index_dir / vectors_file).as_posix()
        if drop_old:
            for file in self.index_dir.glob(f"{self.index_file_name}*"):
                os.remove(file)

        if self.index_type == "DISKANN":
            self.config["data_path"] = self.vectors_file
            self.config["index_prefix"] = self.index_file_path

        self.index = None
        self.bitset = None

    @classmethod
    def config_cls(cls) -> Type[DBConfig]:
        return KnowhereConfig

    @classmethod
    def case_config_cls(cls, index_type: IndexType | None = None) -> Type[DBCaseConfig]:
        return KnowhereIndexConfig

    @contextmanager
    def init(self) -> None:
        import knowhere

        knowhere.SetBuildThreadPool(self.build_threads)
        knowhere.SetSearchThreadPool(self.search_threads)

        index = knowhere.CreateIndex(self.index_type, self.version)

        index_files = self.index_dir.glob(f"{self.index_file_name}*")
        index_exsited = len(list(index_files)) >= 1

        if index_exsited:
            if self.index_type == "DISKANN":
                log.info(
                    f"[DISKANN] Index file existed; Load the index file and Deserialize; {self.index_file_path}"
                )
                index.Deserialize(knowhere.GetBinarySet(),
                                  json.dumps(self.config))
            else:
                log.info(
                    f"[{self.index_type}] Index file existed; Load the index file and Deserialize; {self.index_file_path}"
                )
                indexBinarySet = knowhere.GetBinarySet()
                knowhere.Load(indexBinarySet, self.index_file_path)
                index.Deserialize(indexBinarySet)

            log.info(f"Index ready")
        else:
            log.info(f"index file not existed.")

        self.index = index

        yield

        self.index = None
        self.bitset = None

    def insert_embeddings(
        self,
        embeddings,
        metadata: list[int],
        **kwargs,
    ) -> (int, Exception):
        import knowhere

        if len(embeddings) == 0:
            return 0, None

        if self.index_type == "DISKANN":
            log.info(f"Dump train vectors to {self.vectors_file}")
            with open(self.vectors_file, "wb") as f:
                f.write(struct.pack("I", len(embeddings)))
                f.write(struct.pack("I", len(embeddings[0])))
                # Writing embedding vectors to the binary file
                for element in embeddings:
                    for value in element:
                        f.write(struct.pack("f", value))
                f.close()

            log.info(f"Start building index with {len(embeddings)} vectors")
            self.config.update(self.case_config.index_param())
            log.info(f"build config: {self.config}")
            index = knowhere.CreateIndex(self.index_type, self.version)
            index.Build(knowhere.GetNullDataSet(), json.dumps(self.config))
            log.info(
                f"Serialize and dump the trained index to {self.index_file_path}")
            index.Serialize(knowhere.GetNullDataSet())
            log.info(f"Dump Finished")

        else:
            log.info(f"Start building index with {len(embeddings)} vectors")

            data = knowhere.ArrayToDataSet(embeddings)
            self.config.update(self.case_config.index_param())
            log.info(
                f"Build config: {self.config}, {self.db_config.get('index_type')}, {self.version}"
            )
            index = knowhere.CreateIndex(
                self.db_config.get("index_type"), self.version)
            index.Build(data, json.dumps(self.config))
            indexBinarySet = knowhere.GetBinarySet()
            log.info(f"Serialize the trained index to BinarySet")
            index.Serialize(indexBinarySet)
            log.info(f"Dump the BinarySet to file - {self.index_file_path}")
            knowhere.Dump(indexBinarySet, self.index_file_path)
            log.info(f"Dump Finished")

        return len(embeddings), None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
    ) -> list[int]:
        raise NotImplementedError

    def search_embeddings(
        self,
        queryData,  # ndarray
        k: int = 100,
        filters: dict | None = None,
    ) -> list[int]:
        import knowhere

        query = knowhere.ArrayToDataSet(queryData)
        self.config.update(self.case_config.search_param())
        self.config["k"] = k
        bitset = (
            self.bitset.GetBitSetView() if self.bitset else knowhere.GetNullBitSetView()
        )
        log.info(f"search config: {self.config}")
        ans, _ = self.index.Search(query, json.dumps(self.config), bitset)
        k_dis, k_ids = knowhere.DataSetToArray(ans)
        return k_ids.tolist()

    def convert_to_bitset(self, valid_ids: list[int]):
        import knowhere

        log.info("Convert valid ids to knowher-bitset")
        rowCount = self.index.Count()
        bitset = knowhere.CreateBitSet(rowCount)
        invalid_ids = set(range(rowCount)) - set(valid_ids)
        for invalid_id in invalid_ids:
            bitset.SetBit(invalid_id)
        self.bitset = bitset
        log.info("Convert Finished")

    def optimize(self):
        pass

    def ready_to_load(self):
        pass
