import traceback
import logging
import subprocess
import os
from . import config

log = logging.getLogger(__name__)


def main():
    log.info(f"all configs: {config().display()}")
    run_cmd()


def run_cmd():
    cmd = [
        "python3",
        f"{os.path.dirname(__file__)}/cmd/run.py",
    ]
    log.info(f"go go go cmd")
    log.debug(f"cmd: {cmd}")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        log.info("exit")
    except Exception as e:
        log.warning(f"exit, err={e}\nstack trace={traceback.format_exc(chain=True)}")


if __name__ == "__main__":
    main()
