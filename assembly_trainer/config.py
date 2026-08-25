"""Config schema, loader, and validation (build plan §1/§8).

Everything tunable in the pipeline lives in one YAML file. This module is
the only place that knows the YAML shape; every other module gets typed,
already-resolved values out of a `Config`/`StepConfig` instance instead of
reading dicts.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised for anything wrong with the config file's contents."""


@dataclass
class AppDefaults:
    confidence_threshold: float = 0.60
    inference_interval_ms: int = 200
    stability_window_frames: int = 10
    stability_pass_ratio: float = 0.7
    assembly_roi_margin_ratio: float = 0.35
    # §7: run detection at the highest resolution the model/hardware can
    # sustain at the target inference rate — start high (960/1280), not
    # YOLO's 640 default. Single source of truth for both training
    # (training/train.py) and live inference (detector.py) so they can't
    # drift into two separate hardcoded literals.
    imgsz: int = 960


@dataclass
class CameraConfig:
    device_index: int = 0
    width: int = 1920
    height: int = 1080
    auto_exposure: bool = True
    warmup_seconds: float = 2.5  # settle time for auto-exposure/white-balance after opening
    workstation_roi: dict[str, float] | None = None  # normalized {x,y,w,h}


@dataclass
class EscalationDefaults:
    tier_1_reference_image_after_seconds: float = 20
    tier_2_reference_video_after_seconds: float = 45
    tier_3_trainer_alert_after_seconds: float = 90
    pause_on_partial_progress: bool = False


@dataclass
class SubStep:
    key: str
    ui_prompt: str
    require_roi_empty_first: bool = False


@dataclass
class StepConfig:
    id: int
    name: str
    requires: dict[str, int]
    type: str = "normal"  # "normal" | "sequential_roi"
    roi: str | None = None
    sub_steps: list[SubStep] = field(default_factory=list)
    reference_image: str | None = None
    reference_video: str | None = None
    # Which named hole (from top-level `hole_template`) a peg must land in for
    # this step to be satisfied — only meaningful for steps 1/2 (see
    # hole_position.py). None = no hole-position check for this step (the
    # existing count/orientation gating is unaffected either way).
    target_hole: str | None = None

    # Per-step overrides; None means "fall back to the app/escalation default".
    confidence_threshold: float | None = None
    inference_interval_ms: int | None = None
    stability_window_frames: int | None = None
    stability_pass_ratio: float | None = None
    assembly_roi_margin_ratio: float | None = None
    escalation_overrides: dict[str, Any] = field(default_factory=dict)

    def resolved_escalation(self, defaults: EscalationDefaults) -> EscalationDefaults:
        merged = {**defaults.__dict__, **self.escalation_overrides}
        return EscalationDefaults(**merged)


@dataclass
class Config:
    model_path: str
    classes: list[str]
    app: AppDefaults
    camera: CameraConfig
    escalation: EscalationDefaults
    steps: list[StepConfig]
    # Second, optional model: corner-keypoint pose detector for block_7x7,
    # used only by steps whose target_hole is set (see hole_position.py).
    # None/missing file = hole-position checking is silently disabled,
    # everything else behaves exactly as without it.
    pose_model_path: str | None = None
    # edge name -> ordered list of {"name": str, "x": float, "y": float},
    # normalized 0-1 in the canonical block rectangle (corner order:
    # top-left=(0,0), top-right=(1,0), bottom-right=(1,1), bottom-left=(0,1)
    # -- see hole_position.py CANONICAL_CORNERS). Hole names must be unique
    # across all edges combined.
    hole_template: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def get_step(self, step_id: int) -> StepConfig:
        for step in self.steps:
            if step.id == step_id:
                return step
        raise KeyError(f"no step with id={step_id}")

    def first_step_id(self) -> int:
        return self.steps[0].id

    def next_step_id(self, step_id: int) -> int | None:
        ids = [s.id for s in self.steps]
        idx = ids.index(step_id)
        return ids[idx + 1] if idx + 1 < len(ids) else None

    # -- per-step tunable resolution (§7/§8: step override, else app default) --
    def resolve(self, step: StepConfig, field_name: str) -> Any:
        step_value = getattr(step, field_name, None)
        if step_value is not None:
            return step_value
        return getattr(self.app, field_name)

    def resolve_escalation(self, step: StepConfig) -> EscalationDefaults:
        return step.resolved_escalation(self.escalation)


def _require(d: dict, key: str, ctx: str) -> Any:
    if key not in d:
        raise ConfigError(f"missing required key '{key}' in {ctx}")
    return d[key]


def _build_step(raw: dict, classes: set[str]) -> StepConfig:
    ctx = f"step id={raw.get('id')!r}"
    step_id = _require(raw, "id", ctx)
    requires = _require(raw, "requires", ctx)
    if not isinstance(requires, dict) or not requires:
        raise ConfigError(f"{ctx}: 'requires' must be a non-empty mapping of class -> count")
    for cls, count in requires.items():
        if cls not in classes:
            raise ConfigError(f"{ctx}: requires unknown class '{cls}' (not in top-level 'classes')")
        if not isinstance(count, int) or count < 1:
            raise ConfigError(f"{ctx}: required count for '{cls}' must be a positive int")

    step_type = raw.get("type", "normal")
    if step_type not in ("normal", "sequential_roi"):
        raise ConfigError(f"{ctx}: unknown type '{step_type}'")

    sub_steps_raw = raw.get("sub_steps", [])
    sub_steps = [
        SubStep(
            key=_require(s, "key", f"{ctx} sub_step"),
            ui_prompt=_require(s, "ui_prompt", f"{ctx} sub_step"),
            require_roi_empty_first=bool(s.get("require_roi_empty_first", False)),
        )
        for s in sub_steps_raw
    ]
    if step_type == "sequential_roi":
        if not raw.get("roi"):
            raise ConfigError(f"{ctx}: type=sequential_roi requires a 'roi' name")
        if len(sub_steps) < 2:
            raise ConfigError(f"{ctx}: type=sequential_roi requires at least 2 'sub_steps'")

    for numeric_field in (
        "confidence_threshold",
        "stability_pass_ratio",
        "assembly_roi_margin_ratio",
    ):
        if numeric_field in raw and not (0 < raw[numeric_field] <= 1):
            raise ConfigError(f"{ctx}: '{numeric_field}' must be in (0, 1]")

    escalation_overrides = dict(raw.get("escalation", {}))

    return StepConfig(
        id=step_id,
        name=_require(raw, "name", ctx),
        requires=dict(requires),
        type=step_type,
        roi=raw.get("roi"),
        sub_steps=sub_steps,
        reference_image=raw.get("reference_image"),
        reference_video=raw.get("reference_video"),
        confidence_threshold=raw.get("confidence_threshold"),
        inference_interval_ms=raw.get("inference_interval_ms"),
        stability_window_frames=raw.get("stability_window_frames"),
        stability_pass_ratio=raw.get("stability_pass_ratio"),
        assembly_roi_margin_ratio=raw.get("assembly_roi_margin_ratio"),
        escalation_overrides=escalation_overrides,
        target_hole=raw.get("target_hole"),
    )


def _build_hole_template(raw: dict) -> dict[str, list[dict[str, Any]]]:
    template_raw = raw.get("hole_template", {})
    if not isinstance(template_raw, dict):
        raise ConfigError("top level: 'hole_template' must be a mapping of edge_name -> list of holes")
    seen_names: set[str] = set()
    template: dict[str, list[dict[str, Any]]] = {}
    for edge_name, holes in template_raw.items():
        if not isinstance(holes, list) or not holes:
            raise ConfigError(f"hole_template.{edge_name}: must be a non-empty list of holes")
        built: list[dict[str, Any]] = []
        for hole in holes:
            ctx = f"hole_template.{edge_name}"
            name = _require(hole, "name", ctx)
            x, y = _require(hole, "x", ctx), _require(hole, "y", ctx)
            if not (0 <= x <= 1) or not (0 <= y <= 1):
                raise ConfigError(f"{ctx}: hole '{name}' x/y must be normalized in [0, 1]")
            if name in seen_names:
                raise ConfigError(f"hole_template: duplicate hole name '{name}' (names must be unique across all edges)")
            seen_names.add(name)
            built.append({"name": name, "x": float(x), "y": float(y)})
        template[edge_name] = built
    return template


def load_config(path: str | Path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top-level YAML must be a mapping")

    classes = _require(raw, "classes", "top level")
    if not isinstance(classes, list) or not classes:
        raise ConfigError("top level: 'classes' must be a non-empty list")

    app = AppDefaults(**raw.get("app", {}))
    camera = CameraConfig(**raw.get("camera", {}))
    escalation = EscalationDefaults(**raw.get("escalation", {}))

    steps_raw = _require(raw, "steps", "top level")
    if not isinstance(steps_raw, list) or not steps_raw:
        raise ConfigError("top level: 'steps' must be a non-empty list")

    steps = [_build_step(s, set(classes)) for s in steps_raw]
    steps.sort(key=lambda s: s.id)
    ids = [s.id for s in steps]
    if ids != list(range(1, len(ids) + 1)):
        raise ConfigError(f"top level: step ids must be a contiguous 1..N sequence, got {ids}")

    hole_template = _build_hole_template(raw)
    known_hole_names = {hole["name"] for holes in hole_template.values() for hole in holes}
    for step in steps:
        if step.target_hole is not None and step.target_hole not in known_hole_names:
            raise ConfigError(
                f"step id={step.id}: target_hole '{step.target_hole}' not found in 'hole_template' "
                f"(known holes: {sorted(known_hole_names)})"
            )

    for defaults, name in ((app, "app"), (escalation, "escalation")):
        for f in ("confidence_threshold", "stability_pass_ratio", "assembly_roi_margin_ratio"):
            if hasattr(defaults, f) and not (0 < getattr(defaults, f) <= 1):
                raise ConfigError(f"{name}.{f} must be in (0, 1]")
    if not (
        escalation.tier_1_reference_image_after_seconds
        < escalation.tier_2_reference_video_after_seconds
        < escalation.tier_3_trainer_alert_after_seconds
    ):
        raise ConfigError("escalation tiers must be strictly increasing: tier_1 < tier_2 < tier_3")

    return Config(
        model_path=_require(raw, "model_path", "top level"),
        classes=list(classes),
        app=app,
        camera=camera,
        escalation=escalation,
        steps=steps,
        pose_model_path=raw.get("pose_model_path"),
        hole_template=hole_template,
    )


def deep_copy_config(cfg: Config) -> Config:
    """Useful for tests that mutate a config without touching the shared default."""
    return copy.deepcopy(cfg)
