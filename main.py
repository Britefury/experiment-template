import click

# Job output will go into the folder `logs_and_output/<JOB_GROUP>`
# Log file will be at `logs_and_output/<JOB_GROUP>/log_<JOB_NAME>.txt`
# Checkpoint at `logs_and_output/<JOB_GROUP>/ckpt.pth`
JOB_GROUP = 'test_job'

@click.command()
@click.option('--job_name', type=str, default='')
@click.option('--dataset', type=str, default='cifar')
@click.option('--num_runs', type=int, default=10)
@click.option('--weight', type=float, default=0.1)
@click.option('--continue_from_check', is_flag=True, default=False)
def test_job(job_name, dataset, num_runs, weight, continue_from_check):
    settings = locals().copy()

    import datetime
    import sys
    import job_output

    if job_name == '':
        job_name = 'test_{}'.format(dataset)

        # Note that you could also use the date and time if you like.
        # import datetime
        # job_name = 'test_{}_{}'.format(dataset, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # continue_job: if True, then this will pick up the log file where we left off and continue writing to it
    #   Do this is your code supports restarting from a checkpoint
    # is_master: This is usefule if you use Torch DistributedDataParallel that uses one process per GPU.
    #   In such cases you only want one process (the master) to write to log files, save checkpoints, etc.
    #   So you only have is_master=True for the first process, use False for the rest
    job_out = job_output.JobOutput(JOB_GROUP, job_name, continue_job=continue_from_check, is_main=True)
    # After this call, output to stdout/stderr will be saved to the log file
    job_out.connect_streams()

    # Print the command line used to launch the experiment as it can be useful to keep a record of this
    # (note this will be saved to the log file)
    print('Command line:')
    print(' '.join(sys.argv))

    # Print the all the settings that were passed to the experiment
    print('Settings:')
    print(', '.join(['{}={}'.format(k, settings[k]) for k in sorted(settings.keys())]))

    # Simple checkpointing example
    checkpoint = None
    if continue_from_check and job_out.checkpoint_exists():
        # We are continuing; read existing checkpoint
        checkpoint = job_out.read_checkpoint('cpu')
    else:
        checkpoint = dict(timestamps=[])

    # Write checkpoint
    checkpoint['timestamps'].append(datetime.datetime.now())
    job_out.write_checkpoint(checkpoint)


if __name__ == '__main__':
    test_job()
