import os
from pathlib import Path
import logging
import logzero


def setup(logdir, loglevel="INFO"):
    logdir = Path(logdir)
    loglevel = getattr(logging, loglevel)
    os.makedirs(logdir, exist_ok=True)
    logfile = logdir / "logs.txt"

    # Setup rotating logfile with 3 rotations, each with a maximum filesize of 1MB:
    logzero.logfile(logfile, maxBytes=1e6, backupCount=3)
    logzero.loglevel(loglevel)
    return logzero.logger

