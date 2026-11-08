import wandb

ENTITY = "kahala_patrick-stanford-university"
PROJECT = "pedestrian-forecast"


def start(job_type: str, enabled: bool = True, **kwargs):
    """`enabled=False` returns a no-op run, so callers log unconditionally."""
    return wandb.init(
        entity=ENTITY,
        project=PROJECT,
        job_type=job_type,
        mode=None if enabled else "disabled",
        **kwargs,
    )


def log_file(run, path, name: str, kind: str):
    artifact = wandb.Artifact(name, type=kind)
    artifact.add_file(str(path))
    run.log_artifact(artifact)


def table(df):
    return wandb.Table(dataframe=df.reset_index())
