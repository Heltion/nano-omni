"""Synchronization mechanisms for the four transfer/compute dependencies."""

from typing import Literal

from pydantic import BaseModel, ConfigDict


class Synchronization(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    host_to_copy: Literal["event", "value32"] = "value32"
    copy_to_host: Literal["event_sync"] = "event_sync"
    copy_to_compute: Literal["event", "value32"] = "value32"
    compute_to_copy: Literal["event", "value32"] = "event"
