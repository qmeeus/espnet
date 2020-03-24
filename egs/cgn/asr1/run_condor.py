#!/usr/bin/env python3
import subprocess
import os
from pathlib import Path
import argparse
import re
import random
import string


def random_string(length):
    charset = string.ascii_letters + string.digits
    return(''.join(random.sample(charset * length, length)))

def memory_type(string):
    if not re.match(r"\d+[MG]?", string):
        raise argparse.ArgumentTypeError
    return string


class CondorJob:

    JOB_DIR = "jobs"
    TEMPLATE_FILE = Path(JOB_DIR, "template.job")

    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--clean", action="store_true")
        parser.add_argument("--interactive", action="store_true")
        parser.add_argument("--verbose", "-v", action="store_true")
        parser.add_argument("--dry-run", action="store_true")

        parser.add_argument("--mean_user", action="store_true")
        parser.add_argument("--ncpus", default=1, type=int)
        parser.add_argument("--mem", default="8G", type=memory_type)
        parser.add_argument("--duration", default="100000", type=int)
        parser.add_argument("--ngpu", default=1, type=int)
        parser.add_argument("--project_root", default=Path(__file__).absolute().parent, type=Path)
        parser.add_argument("command", type=str)
        parser.add_argument("--output_dir", default="exp", type=Path)
        parser.add_argument("--exp-name", type=str, required=True)
        parser.add_argument("--tag", default="debug", type=str)
        parser.add_argument("--njobs", default=1, type=int)

        parser.add_argument("--min-cuda-mem", default=None, type=int)
        parser.add_argument("--min-cuda-cap", default=None, type=str)
        parser.add_argument("--singularity", action="store_true")

        options, unknown = parser.parse_known_args()
        options = vars(options)
        options["command_args"] = " ".join(unknown)
        return cls(options)

    def __init__(self, options):
        self.options = options
        self._set_args()
        self._format_args()
        self.jobfile = Path(self.JOB_DIR, f".{random_string(12)}.tmp")
        with open(self.TEMPLATE_FILE) as template, open(self.jobfile, 'w') as jobfile:
            job = template.read().format(**self.options)
            if self.verbose:
                print(job)
            jobfile.write(job)

        self._process = None
        self.run()

        if self.clean:
            if self._process is not None:
                self._process.wait()
            os.remove(self.jobfile)

    def _set_args(self):
        self.clean = self.options.pop("clean")
        self.interactive = self.options.pop("interactive")
        self.verbose = self.options.pop("verbose")
        self.dry_run = self.options.pop("dry_run")
        output_dir = self.options.pop("output_dir")
        exp_name = self.options["exp_name"]
        tag = self.options.get("tag", None)
        self.output_dir = Path(output_dir, exp_name + (f"_{tag}" if tag else ""))

    def run(self):

        command = ["condor_submit", str(self.jobfile)]

        if self.verbose or self.dry_run:
            print("Output dir:", self.output_dir)
            print("Exec:", " ".join(command))

        if self.dry_run:
            return

        os.makedirs(self.output_dir, exist_ok=True)
        
        if self.interactive:
            command.append("-interactive")
            os.system(" ".join(command))
        else:
            self._process = subprocess.Popen(command)

    def _format_constraints(self):

        templates = {
            "min_cuda_mem": "CUDAGlobalMemoryMB > {}",
            "min_cuda_cap": "CUDACapability >= {}",
            "singularity": "HasSingularity"
        }

        constraints = []
        for option_name, template in templates.items():
            option = self.options.pop(option_name, None)
            if option:
                constraints.append(template.format(option))

        self.options["constraints"] = " && ".join(map("( {} )".format, constraints))

    def _format_args(self):

        nice_user = not(self.options.pop("mean_user"))
        if self.interactive:
            nice_user = False
            self.options["duration"] = min(14000, self.options["duration"])

        self.options["nice_user"] = str(nice_user).lower()
        constraints = self._format_constraints()

        exp_name, tag = (self.options.pop(opt_name) for opt_name in ("exp_name", "tag"))
        for opt, ext in zip(("logfile", "stdout", "stderr"), ("log", "out", "err")):
            self.options[opt] = Path(self.output_dir, f"condor.{ext}")

        self.options["command_args"] += f" --tag {tag} --output_dir {self.output_dir} --ngpu {self.options['ngpu']}"


if __name__ == '__main__':
    condor_job = CondorJob.parse_args()

