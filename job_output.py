from typing import Any, Optional, IO
import datetime
import pathlib
import sys
import torch


class LogAlreadyExistsError(Exception):
    pass


class _LogHelper(object):
    def __init__(self, path: pathlib.Path, stream: IO):
        self.path = path
        self.stream = stream

    def write(self, x: str):
        with self.path.open('a+') as f_out:
            f_out.write(x)
        self.stream.write(x)

    def flush(self):
        self.stream.flush()


class _LogSuppress(object):
    def __init__(self, stream: IO):
        self.stream = stream

    def write(self, x: str):
        pass

    def flush(self):
        pass


class JobOutput(object):
    def __init__(self, job_group: str, job_name: str, continue_job: bool = False, is_main: bool = True):
        """Job output for writing output to a log file and saving checkpoints.

        Note that you can disable logging and job output by giving the value `''` (empty string) or `'none'`
        as the `job_name` parameter.

        Creates the following output files:
        - Log file at: `logs_and_output/<job_group>/log_<job_name>.txt`
        - Checkpoint file at: `logs_and_output/<job_group>/<job_name>/ckpt.pth`

        The `logs_and_output/<job_group>/<job_name>` directory is only created if you save a checkpoint
        by calling the `write_checkpoint` method.

        :param job_group: job group name; jobs in the same group have their
        :param job_name: job name
        :param continue_job: use to indicate if you are loading a checkpoint for the purpose of continuing
            the experiment. If False and if the log file already exists, the `LogAlreadyExistsError` exception
            will be raised to prevent you from overwriting results or repeating the experiment.
        :param is_main: If you are using Torch `DistributedDataParallel` to train on multiple GPUs
            with one process per GPU, you only want to write logs and checkpoints from the first (main)
            process. Use `is_main=True` on the first process and `is_main=False` on the others.
        """
        self.is_main = is_main
        output_dir = pathlib.Path('logs_and_output') / job_group
        output_dir.mkdir(parents=True, exist_ok=True)

        self.enabled = job_name not in {'', 'none'}

        if self.enabled:
            # Setup paths
            self.log_path = output_dir / 'log_{}.txt'.format(job_name)
            self.job_dir = output_dir / job_name
            self.checkpoint_path = self.job_dir / 'ckpt.pth'

            # If we are not continuing and the log file exists, throw an exception as the job
            # has already been run
            if is_main:
                if self.log_path.exists():
                    if continue_job:
                        intro_lines = [
                            '=====\n'
                            'Continuing job {}/{} at {}\n'.format(
                                job_group, job_name, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        ]
                    else:
                        raise LogAlreadyExistsError('Log file {} already exists.'.format(self.log_path))
                else:
                    intro_lines = [
                        'Starting job {}/{} at {}\n'.format(
                            job_group, job_name, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                    ]
                self.log_path.open('a+').writelines(intro_lines)

            if self.is_main:
                # Main process: write logs
                self.__stdout = _LogHelper(self.log_path, sys.stdout)
                self.__stderr = _LogHelper(self.log_path, sys.stderr)
            else:
                # Non-main process: don't write logs to the log file and don't send to std-out either
                self.__stdout = _LogSuppress(sys.stdout)
                self.__stderr = _LogSuppress(sys.stderr)
        else:
            self.log_path = None
            self.job_dir = None
            self.checkpoint_path = None
            self.__stdout = None
            self.__stderr = None

    def connect_streams(self):
        """Hook into the stdout and stderr streams, copying their output to a file"""
        if self.__stdout is not None:
            sys.stdout = self.__stdout
            sys.stderr = self.__stderr

    def disconnect_streams(self):
        """Restore the stdout and stderr streams to the way they were before calling `connect_streams`"""
        if self.__stdout is not None:
            sys.stdout = self.__stdout.stream
            sys.stderr = self.__stderr.stream

    def checkpoint_exists(self) -> bool:
        """Determine if a checkpoint file for this experiment exists.

        :return: return True if the checkpoint file exists
        """
        if self.checkpoint_path is not None:
            return self.checkpoint_path.is_file()
        else:
            return False

    def read_checkpoint(self, map_location=None) -> Optional[Any]:
        """Read a checkpoint file from this experiment's checkpoint file.

        :param map_location: (Torch) Device onto which to load the checkpoint
        :return: The checkpoint if the file exists, else None
        """
        if self.checkpoint_path is not None and self.checkpoint_exists():
            return torch.load(self.checkpoint_path, map_location=map_location)
        else:
            return None

    def write_checkpoint(self, data: Any):
        """Write a checkpoint to the checkpoint

        :param data: The data to write to the checkpoint
        """
        # Only write checkpoints from the main process
        if self.checkpoint_path is not None and self.is_main:
            self.job_dir.mkdir(parents=True, exist_ok=True)
            # Write to a temporary new path, in case the job gets terminated during writing
            new_path = self.checkpoint_path.parent / (self.checkpoint_path.name + '.new')
            with new_path.open('wb') as f_ckpt:
                torch.save(data, f_ckpt)
            # Remove the old one if it exists
            if self.checkpoint_path.exists():
                self.checkpoint_path.unlink()
            # Rename
            new_path.rename(self.checkpoint_path)

    def get_output_file_path(self, filename: str) -> Optional[pathlib.Path]:
        """Get the path of a file in the job output directory named `filename`.
        Location will be at `logs_and_output/<job_group>/<job_name>/<filename>`

        :param filename: output filename
        :return: the path as a `pathlib.Path` if output is enabled, None otherwise.
        """
        if self.job_dir is not None:
            self.job_dir.mkdir(parents=True, exist_ok=True)
            return self.job_dir / filename
        else:
            return None
