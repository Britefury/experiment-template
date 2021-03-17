# Simple experiment template

A simple experiment template that I use for deep learning experiments

The `job_output` module copies stdout and stderr to a log file.
If the log file already exists it will raise an exception, preventing you from overwriting results,
unless you are attempting to continue from a checkpoint.

The example experiment in `main` saves the command line arguments and settings to the log file, recording them for
future use.
