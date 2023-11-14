import logging
from contextlib import contextmanager
import struct
from typing import Type
from ..api import VectorDB, DBCaseConfig, DBConfig, IndexType
from .config import KnowhereCloudConfig, KnowhereCloudIndexConfig
import pathlib
import json

log = logging.getLogger(__name__)


class KnowhereCloud(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: KnowhereCloudIndexConfig,
        drop_old: bool = False,
        name: str = "Knowhere-Cloud",
        tmp_dir_path="./results/tmp_knowhere",
        vectors_file="train_vectors.fbin",
        **kwargs,
    ):
        self.name = name
        self.db_config = db_config
        self.case_config = db_case_config
        self.dim = dim
        # self.config = self.db_config.get("config") + f', "dim": {dim}'
        self.config = json.loads(f'{{{self.db_config.get("config")}}}')
        self.config["dim"] = dim

        import knowhere

        self.version = knowhere.GetCurrentVersion()
        self.index = None
        self.bitset = None
        indexFile = (
            self.db_config.get("index_type")
            + "_"
            + db_config.get("config")
            .replace('"', "")
            .replace(" ", "")
            .replace(":", "_")
            .replace(",", "_")
        )
        tmp_dir = pathlib.Path(tmp_dir_path)
        if not tmp_dir.exists():
            tmp_dir.mkdir(parents=True)
        self.indexFile = (tmp_dir / indexFile).as_posix()
        self.vectors_file = (tmp_dir / vectors_file).as_posix()

    @classmethod
    def config_cls(cls) -> Type[DBConfig]:
        return KnowhereCloudConfig

    @classmethod
    def case_config_cls(cls, index_type: IndexType | None = None) -> Type[DBCaseConfig]:
        return KnowhereCloudIndexConfig

    @contextmanager
    def init(self) -> None:
        import knowhere

        index = knowhere.CreateIndex(self.db_config.get("index_type"), self.version)
        filePath = pathlib.Path(self.indexFile + "_mem.index.bin")
        if filePath.exists():
            log.info(
                f"Index file existed; Load the index file and Deserialize; {self.indexFile}"
            )
            self.config.update(self.case_config.index_param())
            self.config["data_path"] = self.vectors_file
            self.config["index_prefix"] = self.indexFile
            index.Deserialize(knowhere.GetBinarySet(), json.dumps(self.config))
            log.info(f"Index ready")
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
        self.config["data_path"] = self.vectors_file
        self.config["index_prefix"] = self.indexFile
        index = knowhere.CreateIndex(self.db_config.get("index_type"), self.version)
        log.info(f"config: {self.config}")
        index.Build(knowhere.GetNullDataSet(), json.dumps(self.config))
        log.info(f"Serialize and dump the trained index to {self.indexFile}")
        index.Serialize(knowhere.GetNullDataSet())
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
        # config = f'{{{self.config}{self.case_config.search_param()}, "k": {k}}}'
        self.config.update(self.case_config.search_param())
        self.config["k"] = k
        bitset = (
            self.bitset.GetBitSetView() if self.bitset else knowhere.GetNullBitSetView()
        )
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
