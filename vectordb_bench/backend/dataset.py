"""
Usage:
    >>> from xxx.dataset import Dataset
    >>> Dataset.Cohere.get(100_000)
"""

import os
import logging
import pathlib
from hashlib import md5
from enum import Enum
import s3fs
import pandas as pd
from tqdm import tqdm
from pydantic import validator, PrivateAttr
import polars as pl
from pyarrow.parquet import ParquetFile

from ..base import BaseModel
from .. import config
from ..backend.clients import MetricType
from . import utils

log = logging.getLogger(__name__)


class BaseDatasetWithDefaultSize(BaseModel):
    name: str
    size: int
    dim: int
    metric_type: MetricType
    use_shuffled: bool
    _size_label: dict = PrivateAttr()
    check_s3: bool = True
    default_dir: str | None = None
    int_column_names: list[str] = ["id"]
    category_column_names: list[str] = []

    @property
    def label(self) -> str:
        return self._size_label.get(self.size)

    @property
    def dir_name(self) -> str:
        return f"{self.name}_{self.label}_{utils.numerize(self.size)}".lower()


class BaseDataset(BaseDatasetWithDefaultSize):
    @validator("size")
    def verify_size(cls, v):
        if v not in cls._size_label:
            raise ValueError(
                f"Size {v} not supported for the dataset, expected: {cls._size_label.keys()}"
            )
        return v


class LAION(BaseDataset):
    name: str = "LAION"
    dim: int = 768
    metric_type: MetricType = MetricType.L2
    use_shuffled: bool = False
    _size_label: dict = {100_000_000: "LARGE"}


class GLOVE_200(BaseDatasetWithDefaultSize):
    name: str = "GLOVE_200"
    dim: int = 200
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = config.USE_SHUFFLED_DATA
    size: int = 1_183_514
    _size_label: dict = {}
    check_s3: bool = False
    default_dir: str = config.NAS_ADDRESS

    @property
    def label(self) -> str:
        return self.name

    @property
    def dir_name(self) -> str:
        return f"{self.name}"


class LASTFM(BaseDatasetWithDefaultSize):
    name: str = "LAST_FM"
    dim: int = 65
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = config.USE_SHUFFLED_DATA
    size: int = 292_385
    _size_label: dict = {}
    check_s3: bool = False
    default_dir: str = config.NAS_ADDRESS

    @property
    def label(self) -> str:
        return self.name

    @property
    def dir_name(self) -> str:
        return f"{self.name}"


class GIST_768(BaseDatasetWithDefaultSize):
    name: str = "GIST_768"
    dim: int = 768
    metric_type: MetricType = MetricType.L2
    use_shuffled: bool = config.USE_SHUFFLED_DATA
    size: int = 1_000_000
    _size_label: dict = {}
    check_s3: bool = False
    default_dir: str = config.NAS_ADDRESS

    @property
    def label(self) -> str:
        return self.name

    @property
    def dir_name(self) -> str:
        return f"{self.name}"


class GIST(BaseDataset):
    name: str = "GIST"
    dim: int = 960
    metric_type: MetricType = MetricType.L2
    use_shuffled: bool = False
    _size_label: dict = {
        100_000: "SMALL",
        1_000_000: "MEDIUM",
    }


class Cohere(BaseDataset):
    name: str = "Cohere"
    dim: int = 768
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = config.USE_SHUFFLED_DATA
    _size_label: dict = {
        100_000: "SMALL",
        1_000_000: "MEDIUM",
        10_000_000: "LARGE",
    }


class Glove(BaseDataset):
    name: str = "Glove"
    dim: int = 200
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = False
    _size_label: dict = {1_000_000: "MEDIUM"}


class SIFT(BaseDataset):
    name: str = "SIFT"
    dim: int = 128
    metric_type: MetricType = MetricType.L2
    use_shuffled: bool = False
    _size_label: dict = {
        500_000: "SMALL",
        5_000_000: "MEDIUM",
        50_000_000: "LARGE",
    }


class OpenAI(BaseDataset):
    name: str = "OpenAI"
    dim: int = 1536
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = config.USE_SHUFFLED_DATA
    _size_label: dict = {
        50_000: "SMALL",
        500_000: "MEDIUM",
        5_000_000: "LARGE",
    }


def category_num_to_column_name(category_num: int):
    return f"scalar-{category_num}"


int_filter_rates = [
    None,
    0.0,
    0.01,
    0.05,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    0.95,
    0.98,
    0.99,
]
category_nums = [2, 5, 10, 100, 1000]


class ScalarDataset(BaseDataset):
    check_s3: bool = False
    default_dir: str = config.NAS_ADDRESS

    int_filter_rates: list[float | None] = int_filter_rates
    category_nums: list[int] = category_nums
    category_column_names: list[str] = [
        category_num_to_column_name(category_num) for category_num in category_nums
    ]


class ScalarCohere(ScalarDataset):
    name: str = "Cohere"
    dim: int = 768
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = config.USE_SHUFFLED_DATA
    size: int = 1_000_000
    _size_label: dict = {
        1_000_000: "MEDIUM",
    }

    @property
    def dir_name(self) -> str:
        return f"{self.name}_{self.label}_{utils.numerize(self.size)}_scalars".lower()


class ScalarOpenAI(ScalarDataset):
    name: str = "OpenAI"
    dim: int = 1536
    metric_type: MetricType = MetricType.COSINE
    use_shuffled: bool = config.USE_SHUFFLED_DATA
    size: int = 500_000
    _size_label: dict = {
        500_000: "MEDIUM",
    }
    check_s3: bool = False
    default_dir: str = config.NAS_ADDRESS

    @property
    def dir_name(self) -> str:
        return f"{self.name}_{self.label}_{utils.numerize(self.size)}_scalars".lower()


class ScalarGIST768(ScalarDataset):
    name: str = "GIST768"
    dim: int = 768
    metric_type: MetricType = MetricType.L2
    use_shuffled: bool = config.USE_SHUFFLED_DATA
    size: int = 1_000_000
    _size_label: dict = {
        1_000_000: "MEDIUM",
    }
    check_s3: bool = False
    default_dir: str = config.NAS_ADDRESS

    @property
    def dir_name(self) -> str:
        return f"{self.name}_{self.label}_{utils.numerize(self.size)}_scalars".lower()


class ScalarText2img(ScalarDataset):
    name: str = "Text2Img"
    dim: int = 200
    metric_type: MetricType = MetricType.IP
    use_shuffled: bool = config.USE_SHUFFLED_DATA
    size: int = 5_000_000
    _size_label: dict = {
        5_000_000: "MEDIUM",
    }
    check_s3: bool = False
    default_dir: str = config.NAS_ADDRESS

    @property
    def dir_name(self) -> str:
        return f"{self.name}_{self.label}_{utils.numerize(self.size)}_scalars".lower()


class DatasetManager(BaseModel):
    """Download dataset if not int the local directory. Provide data for cases.

    DatasetManager is iterable, each iteration will return the next batch of data in pandas.DataFrame

    Examples:
        >>> cohere = Dataset.COHERE.manager(100_000)
        >>> for data in cohere:
        >>>    print(data.columns)
    """

    data: BaseDatasetWithDefaultSize
    # test_data: pd.DataFrame | None = None
    train_files: list[str] = []

    def __eq__(self, obj):
        if isinstance(obj, DatasetManager):
            return self.data.name == obj.data.name and self.data.label == obj.data.label
        return False

    @property
    def data_dir(self) -> pathlib.Path:
        """data local directory: config.DATASET_LOCAL_DIR/{dataset_name}/{dataset_dirname}

        Examples:
            >>> sift_s = Dataset.SIFT.manager(500_000)
            >>> sift_s.relative_path
            '/tmp/vectordb_bench/dataset/sift/sift_small_500k/'
        """
        if self.data.default_dir is not None:
            return pathlib.Path(self.data.default_dir, self.data.dir_name)

        return pathlib.Path(
            config.DATASET_LOCAL_DIR, self.data.name.lower(), self.data.dir_name.lower()
        )

    @property
    def download_dir(self) -> str:
        """data s3 directory: config.DEFAULT_DATASET_URL/{dataset_dirname}

        Examples:
            >>> sift_s = Dataset.SIFT.manager(500_000)
            >>> sift_s.download_dir
            'assets.zilliz.com/benchmark/sift_small_500k'
        """
        return f"{config.DEFAULT_DATASET_URL}{self.data.dir_name}"

    def __iter__(self):
        return DataSetIterator(self)

    def _validate_local_file(self):
        if not self.data_dir.exists():
            log.info(f"local file path not exist, creating it: {self.data_dir}")
            self.data_dir.mkdir(parents=True)

        fs = s3fs.S3FileSystem(anon=True, client_kwargs={"region_name": "us-west-2"})
        dataset_info = fs.ls(self.download_dir, detail=True)
        if len(dataset_info) == 0:
            raise ValueError(f"No data in s3 for dataset: {self.download_dir}")
        path2etag = {info["Key"]: info["ETag"].split('"')[1] for info in dataset_info}

        perfix_to_filter = "train" if self.data.use_shuffled else "shuffle_train"
        filtered_keys = [
            key
            for key in path2etag.keys()
            if key.split("/")[-1].startswith(perfix_to_filter)
        ]
        for k in filtered_keys:
            path2etag.pop(k)

        # get local files ended with '.parquet'
        file_names = [p.name for p in self.data_dir.glob("*.parquet")]
        log.info(f"local files: {file_names}")
        log.info(f"s3 files: {path2etag.keys()}")
        downloads = []
        if len(file_names) == 0:
            log.info("no local files, set all to downloading lists")
            downloads = path2etag.keys()
        else:
            # if local file exists, check the etag of local file with s3,
            # make sure data files aren't corrupted.
            for name in tqdm([key.split("/")[-1] for key in path2etag.keys()]):
                s3_path = f"{self.download_dir}/{name}"
                local_path = self.data_dir.joinpath(name)
                log.debug(f"s3 path: {s3_path}, local_path: {local_path}")
                if not local_path.exists():
                    log.info(
                        f"local file not exists: {local_path}, add to downloading lists"
                    )
                    downloads.append(s3_path)

                elif not self.match_etag(path2etag.get(s3_path), local_path):
                    log.info(
                        f"local file etag not match with s3 file: {local_path}, add to downloading lists"
                    )
                    downloads.append(s3_path)

        for s3_file in tqdm(downloads):
            log.debug(f"downloading file {s3_file} to {self.data_dir}")
            fs.download(s3_file, self.data_dir.as_posix())

    def match_etag(self, expected_etag: str, local_file) -> bool:
        """Check if local files' etag match with S3"""

        def factor_of_1MB(filesize, num_parts):
            x = filesize / int(num_parts)
            y = x % 1048576
            return int(x + 1048576 - y)

        def calc_etag(inputfile, partsize):
            md5_digests = []
            with open(inputfile, "rb") as f:
                for chunk in iter(lambda: f.read(partsize), b""):
                    md5_digests.append(md5(chunk).digest())
            return md5(b"".join(md5_digests)).hexdigest() + "-" + str(len(md5_digests))

        def possible_partsizes(filesize, num_parts):
            return (
                lambda partsize: partsize < filesize
                and (float(filesize) / float(partsize)) <= num_parts
            )

        filesize = os.path.getsize(local_file)
        le = ""
        if "-" not in expected_etag:  # no spliting uploading
            with open(local_file, "rb") as f:
                le = md5(f.read()).hexdigest()
                log.debug(f"calculated local etag {le}, expected etag: {expected_etag}")
                return expected_etag == le
        else:
            num_parts = int(expected_etag.split("-")[-1])
            partsizes = [  ## Default Partsizes Map
                8388608,  # aws_cli/boto3
                15728640,  # s3cmd
                factor_of_1MB(
                    filesize, num_parts
                ),  # Used by many clients to upload large files
            ]

            for partsize in filter(possible_partsizes(filesize, num_parts), partsizes):
                le = calc_etag(local_file, partsize)
                log.debug(f"calculated local etag {le}, expected etag: {expected_etag}")
                if expected_etag == le:
                    return True
        return False

    def prepare(self, check=True) -> bool:
        """Download the dataset from S3
        url = f"{config.DEFAULT_DATASET_URL}/{self.data.dir_name}"

        download files from url to self.data_dir, there'll be 4 types of files in the data_dir
            - train*.parquet: for training
            - test.parquet: for testing
            - neighbors.parquet: ground_truth of the test.parquet
            - neighbors_head_1p.parquet: ground_truth of the test.parquet after filtering 1% data
            - neighbors_99p.parquet: ground_truth of the test.parquet after filtering 99% data
        """
        if check and self.data.check_s3:
            self._validate_local_file()

        prefix = "shuffle_train" if self.data.use_shuffled else "train"
        self.train_files = sorted(
            [f.name for f in self.data_dir.glob(f"{prefix}*.parquet")]
        )
        log.debug(f"{self.data.name}: available train files {self.train_files}")
        # self.test_data = self._read_file("test.parquet")
        return True
    
    def get_test_data(self):
        return self._read_file("test.parquet")

    def check_case_has_groundtruth(self, groundtruth_file: str) -> bool:
        if (
            len(groundtruth_file) > 0
            and pathlib.Path(self.data_dir, groundtruth_file).exists()
        ):
            return True
        else:
            return False

    def get_ground_truth(self, groundtruth_file: str) -> pd.DataFrame:
        return self._read_file(groundtruth_file)

    def _read_file(self, file_name: str) -> pd.DataFrame:
        """read one file from disk into memory"""
        log.info(f"Read the entire file into memory: {file_name}")
        p = pathlib.Path(self.data_dir, file_name)
        if not p.exists():
            log.warning(f"No such file: {p}")
            return pd.DataFrame()

        return pl.read_parquet(p)


class DataSetIterator:
    def __init__(self, dataset: DatasetManager):
        self._ds = dataset
        self._idx = 0  # file number
        self._cur = None
        self._sub_idx = [
            0 for i in range(len(self._ds.train_files))
        ]  # iter num for each file

    def _get_iter(self, file_name: str):
        p = pathlib.Path(self._ds.data_dir, file_name)
        log.info(f"Get iterator for {p.name}")
        if not p.exists():
            raise IndexError(f"No such file {p}")
            log.warning(f"No such file: {p}")
        return ParquetFile(p).iter_batches(config.NUM_PER_BATCH)

    def __next__(self) -> pd.DataFrame:
        """return the data in the next file of the training list"""
        if self._idx < len(self._ds.train_files):
            if self._cur is None:
                file_name = self._ds.train_files[self._idx]
                self._cur = self._get_iter(file_name)

            try:
                return next(self._cur).to_pandas()
            except StopIteration:
                if self._idx == len(self._ds.train_files) - 1:
                    raise StopIteration from None

                self._idx += 1
                file_name = self._ds.train_files[self._idx]
                self._cur = self._get_iter(file_name)
                return next(self._cur).to_pandas()
        raise StopIteration


class Dataset(Enum):
    """
    Value is Dataset classes, DO NOT use it
    Example:
        >>> all_dataset = [ds.name for ds in Dataset]
        >>> Dataset.COHERE.manager(100_000)
        >>> Dataset.COHERE.get(100_000)
    """

    LAION = LAION
    GIST = GIST
    COHERE = Cohere
    GLOVE = Glove
    SIFT = SIFT
    OPENAI = OpenAI
    GLOVE_200 = GLOVE_200
    LASTFM = LASTFM
    GIST_768 = GIST_768
    ScalarCohere = ScalarCohere
    ScalarOpenAI = ScalarOpenAI
    ScalarGIST768 = ScalarGIST768
    ScalarText2img = ScalarText2img

    def get(self, size: int | None = None) -> BaseDatasetWithDefaultSize:
        return self.value(size=size) if size is not None else self.value()

    def manager(self, size: int | None = None) -> DatasetManager:
        return DatasetManager(data=self.get(size))


class ScalarDatasetLabel(Enum):
    Cohere = "cohere"
    Openai = "openai"
    Text2img = "text2img"
    Gist768 = "gist768"

    def get_dataset(self) -> DatasetManager:
        if self not in scalarDatasetLabel2dataset:
            raise RuntimeError(f"Invalid ScalarDatasetLabel - {self}")
        return scalarDatasetLabel2dataset.get(self)


scalarDatasetLabel2dataset = {
    ScalarDatasetLabel.Cohere: Dataset.ScalarCohere.manager(),
    ScalarDatasetLabel.Openai: Dataset.ScalarOpenAI.manager(),
    ScalarDatasetLabel.Text2img: Dataset.ScalarText2img.manager(),
    ScalarDatasetLabel.Gist768: Dataset.ScalarGIST768.manager(),
}
