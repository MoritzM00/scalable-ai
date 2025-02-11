# Scalable AI

![Tests](https://img.shields.io/github/actions/workflow/status/MoritzM00/scalable-ai/test_deploy.yaml?style=for-the-badge&label=Test%20and%20Deploy)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=for-the-badge)][pre-commit]
![License](https://img.shields.io/github/license/MoritzM00/Scalable AI?style=for-the-badge)

[pre-commit]: https://github.com/pre-commit/pre-commit

---

## Working with a Slurm cluster

In this project, we are working on the bwUniCluster 2.0 located at the KIT in Karlsruhe.
[Hardware and Architecture](https://wiki.bwhpc.de/e/BwUniCluster2.0/Hardware_and_Architecture)

### Viewing idle resources.
Execute
```bash
sinfo_t_idle
```
to see idle resources.

### Running a job
Either use `sbatch <script>` to queue a batch job or run
 `salloc` to interactively run a job on a node.

[Batch Queues](https://wiki.bwhpc.de/e/BwUniCluster2.0/Batch_Queues)

### Monitoring a job
Use `squeue` to see the status of your jobs.

Use `scontrol show job <jobid>` to see detailed information about a job.
See [here](https://wiki.bwhpc.de/e/BwUniCluster2.0/Slurm#Detailed_job_information_:_scontrol_show_job) for more information.

### Cancelling a job

Use `scancel <jobid>` to cancel a job.



## Quick Start

Below you can find the quick start guide for development.
### Set up the environment

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Set up the environment:

```bash
make setup
source .venv/bin/activate
```
### Install new packages

To install new PyPI packages, run:

```bash
uv add <package-name>
```

To add dev-dependencies, run:

```bash
uv add --dev <package-name>
```

### Documentation

The Documentation is automatically deployed to GitHub Pages.

To view the documentation locally, run:

```bash
make docs_view
```

## Credits

This project was generated with the [Light-weight Python Template](https://github.com/MoritzM00/python-template) by Moritz Mistol.
