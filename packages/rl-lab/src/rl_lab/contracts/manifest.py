"""Experiment manifest (schema 1.x): what an experiment is, how it was trained, and which representation views it has.

The manifest is the only source of artifact roles: files it does not mention are ignored.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"
SUPPORTED_MAJOR = 1


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Exporter(Model):
    name: str
    version: str
    source_revision: str | None = None


class Algorithm(Model):
    name: str
    revision: str | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)


class TaskDefinition(Model):
    """What the environment's rewards and terminations actually are, so runs under one environment name but with
    different task code can be told apart."""

    source: str
    fingerprint: str
    note: str | None = None


class Environment(Model):
    id: str
    tasks: list[str] = Field(default_factory=list)
    task_definition: TaskDefinition | None = None


class TrainingData(Model):
    kind: Literal["dataset", "self_collected", "replay"]
    id: str | None = None
    transitions: int | None = None


class Training(Model):
    mode: Literal["offline", "online"]
    objective: Literal["reward_free", "task_reward"]
    trained_task: str | None = None
    data: TrainingData
    # Explicit units: gradient_steps, env_steps (environment/physics steps), agent_steps (decisions).
    budget: dict[str, int | None] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _task_reward_needs_a_task(self) -> Training:
        if self.objective == "task_reward" and not self.trained_task:
            raise ValueError("objective 'task_reward' requires trained_task")
        return self


class Experiment(Model):
    id: str
    group: str
    seed: int
    algorithm: Algorithm
    environment: Environment
    training: Training


class DatasetRef(Model):
    id: str
    hash: str


class Protocol(Model):
    """How a view was computed from the dataset's inputs."""

    input: str                                   # e.g. final_state_and_action, window_filter
    window_length: int | None = None
    conditioning: dict[str, Any] = Field(default_factory=lambda: {"type": "none"})
    stochastic: bool = False
    posterior: Literal["sample", "mode"] | None = None
    sampling_noise: dict[str, float | None] | None = None
    note: str | None = None


class View(Model):
    """One named representation of the experiment on one diagnostic dataset at one checkpoint."""

    id: str
    role: Literal["trained", "control"]
    control_kind: Literal["untrained", "observation", "random_projection"] | None = None
    control_of: str | None = None
    checkpoint_step: int
    checkpoint_unit: Literal["gradient_steps", "env_steps", "agent_steps"]
    dataset: DatasetRef
    path: str
    protocol: Protocol

    @model_validator(mode="after")
    def _control_fields(self) -> View:
        if self.role == "control" and (self.control_kind is None or self.control_of is None):
            raise ValueError(f"view {self.id!r}: control views need control_kind and control_of")
        if self.role == "trained" and (self.control_kind is not None or self.control_of is not None):
            raise ValueError(f"view {self.id!r}: trained views cannot have control_kind/control_of")
        return self


class Artifact(Model):
    kind: Literal["training_metrics", "evaluation_metrics", "task_latents", "provenance"]
    path: str
    description: str | None = None


class Manifest(Model):
    schema_version: str
    exporter: Exporter
    experiment: Experiment
    views: list[View] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> Manifest:
        major = self.schema_version.split(".")[0]
        if not major.isdigit() or int(major) != SUPPORTED_MAJOR:
            raise ValueError(f"unsupported schema_version {self.schema_version!r}; this lab reads {SUPPORTED_MAJOR}.x")
        ids = [view.id for view in self.views]
        duplicates = sorted({view_id for view_id in ids if ids.count(view_id) > 1})
        if duplicates:
            raise ValueError(f"duplicate view ids: {duplicates}")
        for view in self.views:
            if view.control_of is not None and view.control_of not in ids:
                raise ValueError(f"view {view.id!r} is a control of unknown view {view.control_of!r}")
        return self

    def view(self, view_id: str) -> View:
        for view in self.views:
            if view.id == view_id:
                return view
        raise KeyError(f"no view {view_id!r} in experiment {self.experiment.id!r}")
