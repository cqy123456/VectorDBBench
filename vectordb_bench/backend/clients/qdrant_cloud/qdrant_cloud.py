"""Wrapper around the QdrantCloud vector database over VectorDB"""

import logging
import time
from contextlib import contextmanager
from typing import Type
import faiss
import psutil
import os
import numpy as np
from ..api import VectorDB, DBConfig, DBCaseConfig, IndexType
from .config import QdrantConfig, QdrantIndexConfig
from qdrant_client.http.models import (
    CollectionStatus,
    VectorParams,
    PayloadSchemaType,
    Batch,
    Filter,
    FieldCondition,
    Range,
)

from qdrant_client import QdrantClient


log = logging.getLogger(__name__)


class QdrantCloud(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "QdrantCloudCollection",
        drop_old: bool = False,
        **kwargs,
    ):
        """Initialize wrapper around the QdrantCloud vector database."""
        print("db_config: ", db_config)

        self._index_key = db_config["index_type"]
        self._search_params = db_config["search_params"]
        self._index_path = self._index_key + ".bin"
        self._train_set = []
        self._train_ids = []
        self._train = True
        self._dim = dim
        self.index = faiss.index_factory(self._dim, self._index_key)
        self.load_mem = 0
        if os.path.exists(self._index_path):
            print("index path exist , loading index from index file")
            process = psutil.Process()
            def load_index(file_name):
                self.index = faiss.read_index(self._index_path)
                return 
            mem_beg = process.memory_info()
            load_index(self._index_path)
            mem_end = process.memory_info()
            rss = (mem_end.rss - mem_beg.rss) / 1024 / 1024
            vms = (mem_end.vms - mem_beg.vms) / 1024 / 1024
            print('Current memory usage after load: RSS=%.2fMB, VMS=%.2fMB' % (rss, vms))
            self.load_mem = rss
            self._train = False

    @classmethod
    def config_cls(cls) -> Type[DBConfig]:
        return QdrantConfig

    @classmethod
    def case_config_cls(cls, index_type: IndexType | None = None) -> Type[DBCaseConfig]:
        return QdrantIndexConfig

    @contextmanager
    def init(self) -> None:
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        if os.path.exists(self._index_path) and self._train:
            print("index path exist , loading index from index file")
            process = psutil.Process()
            mem_beg = process.memory_info()
            self.index = faiss.read_index(self._index_path)
            mem_end = process.memory_info()
            rss = (mem_end.rss - mem_beg.rss) / 1024 / 1024
            vms = (mem_end.vms - mem_beg.vms) / 1024 / 1024
            print('Current memory usage after load: RSS=%.2fMB, VMS=%.2fMB' % (rss, vms))
            self._train = False
        if self._index_key.find("HNSW") != -1:
            faiss.ParameterSpace().set_index_parameter(self.index, "efSearch", self._search_params)
    
        if self._index_key.find("IVF") != -1:
            faiss.ParameterSpace().set_index_parameter(self.index, "nprobe", self._search_params)
        yield

    def ready_to_load(self):
        return 

    def optimize(self):
        pass

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> (int, Exception):
        assert self.index != None
        if (self._train == False):
            return len(metadata), None
        for i in range(len(embeddings)):
            self._train_set.append(np.array(embeddings[i]))
            self._train_ids.append(metadata[i])
        if kwargs.get("last_batch"):
            print("begin to train faiss index", type(self._train_set))
            self._train_set = np.array(self._train_set)
            self.index.train(self._train_set)
            print("add data into faiss index")
            self.index.add(self._train_set)
            self._train_set = None
            print("write faiss index to " + self._index_path)
            faiss.write_index(self.index, self._index_path)
        return len(metadata), None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        """Perform a search on a query embedding and return results with score.
        Should call self.init() first.
        """
        D, I = self.index.search(np.array(query).reshape((1, self._dim)), k = k)
        
        return [i for i in I[0]]
    
    def search_batch(
        self,
        query: list[list[float]],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        """Perform a search on a query embedding and return results with score.
        Should call self.init() first.
        """
        D, I = self.index.search(np.stack(query), k = k)
        return I