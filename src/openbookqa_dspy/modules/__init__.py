from __future__ import annotations

"""Module namespace for approach types and classes."""

from enum import Enum


class ApproachEnum(str, Enum):
    baseline = "baseline"
    mipro = "mipro"
