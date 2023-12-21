import json
import sys
from vectordb_bench.cmd.knowhere_ci.run import get_knowhere_test_results


def dump_result(fn):
    data = get_knowhere_test_results()
    with open(fn, 'w') as f:
        json.dump(data, f)
    f.close()


if __name__ == "__main__":
    file_name = sys.argv[1]
    dump_result(file_name)

