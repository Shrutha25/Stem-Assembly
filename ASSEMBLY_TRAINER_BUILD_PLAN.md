# Assembly Trainer — Build Plan

A continuous-vision training-station app: a fixed camera watches a trainee build a
physical assembly across 10 steps. The system auto-detects step completion (no
button press), gives tiered escalating help if a trainee is stuck, and flags a
human trainer if they're still stuck after that.

This document is the full spec — hand it to an engineer or an AI coding agent to
build from scratch.

---

## 1. Core concept

- One trainee, one fixed camera, one physical fixture/workstation.
- **Continuous auto-detection** — no "Check" button. The system watches
  continuously and infers step completion on its own.
- Steps are a **simple linear sequence** (not a DAG) — strict order, step N+1
  only becomes "current" once step N is confirmed complete.
- On getting stuck, a **3-tier escalation** kicks in automatically based on
  elapsed time without progress: reference image → reference video → trainer
  alert.
- All tunables (thresholds, timings, classes, messages) live in **one config
  file** — no hardcoded values scattered through the pipeline.

---

## 2. Architecture

**Detection + tracking + rule-engine pipeline**, not per-frame whole-image
classification:

```
Camera frame
  → YOLO object detector (parts/tools/states)
  → ByteTrack (temporal association / persistent IDs, for stable counting)
  → per-step class gate (only trust detections relevant to current step)
  → stability window (majority vote over recent frames, 3-way outcome)
  → rule engine / state machine (step requirements met? advance)
  → escalation timer (tiered help if stuck)
  → UI feedback
```

### Why detection, not classification
Steps differ by *which small parts are present/in what state*, not by whole-scene
gestalt. A detector gives per-object, interpretable signals ("screw not
detected" vs. an opaque "wrong" label) and is far more robust to pose/lighting
variation than a global image classifier.

### Why tracking
Needed for **stable counting** (e.g. step 3 needs "5 more short pegs" — count
matters, not identity) and to survive brief occlusion (a hand passing over a
part shouldn't reset detection state). Full re-identification is *not* needed —
parts within a class are visually identical, so this is a counting/stability
problem, not an instance-identity problem.

### Three-way classification outcome
Every per-frame read for a required class is `correct` / `incorrect` /
`uncertain` (softmax confidence below a per-step threshold → `uncertain`).
`uncertain` frames are **excluded** from the stability window entirely — they
don't count for or against. This matters a lot more here than in a
press-to-Check system, since continuous detection sees far more mid-motion /
occluded / blurry frames.

---

## 3. Parts & classes

```yaml
classes:
  - short_peg
  - long_peg
  - block_7x14
  - block_7x7
  - castor_wheel
  - motor_lead_loose        # motor state, not a separate object from motor_lead_connected
  - motor_lead_connected
  - axle
  - wheel
  - peecee
```

**Known high-risk confusion pairs** (prioritize labeling diversity + resolution
here):
- `short_peg` vs `long_peg`
- `block_7x7` vs `block_7x14`
- `motor_lead_loose` vs `motor_lead_connected` — highest risk, see §5 (step 10).

---

## 4. Step sequence (locked, linear)

```yaml
step_1:  {block_7x14: 1, short_peg: 2}
step_2:  {block_7x14: 1, short_peg: 2, block_7x7: 1}
step_3:  {block_7x14: 1, short_peg: 7, block_7x7: 1}
step_4:  {..step_3.., long_peg: 2}
step_5:  {..step_4.., motor_lead_loose: 2}
step_6:  {..step_5.., axle: 2}
step_7:  {..step_6.., wheel: 2}
step_8:  {..step_7.., castor_wheel: 1}
step_9:  {..step_8.., peecee: 1}
step_10: {..step_9.., motor_lead_connected: 2}   # sequential ROI sub-flow, see §5
```

Each step's requirement should be checked as **placement within an ROI
relative to the growing assembly's detected bounding box**, not mere presence
anywhere in frame (a part still sitting in a parts bin shouldn't count).

---

## 5. Step 10 — special case (lead-state, single ROI, sequential)

Motors are wired to the PeeCee. `motor_lead_loose` vs `motor_lead_connected`
cannot be told apart in a normal wide shot (verified: not distinguishable even
by eye at that scale) — solved with a **single fixed ROI + UI-guided sequential
flow**, not simultaneous dual-ROI, and not a `wire` class.

```yaml
step_10:
  roi: {region: "peecee_connector_zone", anchor: "relative to detected peecee bbox"}
  classify: connector_state   # loose vs connected, this ROI only

  sub_step_a:
    ui_prompt: "Show first motor connection"
    wait_for: roi.connector_state == connected (stable over window)
    on_confirm: side_1_done = true

  sub_step_b:
    ui_prompt: "Show second motor connection"
    require: roi goes empty/reset before this sub-step starts listening   # anti-gaming
    wait_for: roi.connector_state == connected (stable over window)
    on_confirm: side_2_done = true

  step_10_complete: side_1_done AND side_2_done
```

Implementation notes:
- The reset-between-confirmations requirement prevents a worker from holding
  one already-connected side in frame and double-counting it as both sides.
- This is the highest-risk classification in the whole system — give it a
  stricter per-step confidence threshold and a longer stability window than
  other steps (see §7).
- UI should visibly show "Step 10 (1/2)" → "(2/2)" progress, not just a spinner.

---

## 6. Interaction & escalation model

No Check button. Timer-based, per-step, resets when the step becomes current.

```yaml
escalation:
  tier_1_reference_image_after_seconds: 20   # default; per-step override allowed
  tier_2_reference_video_after_seconds: 45
  tier_3_trainer_alert_after_seconds: 90
```

Rules:
- Timer starts when a step becomes "current" (previous step just confirmed).
- Tiers **stack** — video stays visible when tier 3 fires; trainer alert is an
  *addition* (notify trainer dashboard/device), not a UI takeover that stops
  the trainee from continuing to try.
- Consider pausing/resetting the timer if the detector sees genuine partial
  progress (some but not all required parts appearing) — avoid punishing a
  trainee who's visibly working correctly but slowly. (Flagged as a nice-to-have;
  simple elapsed-time-only is an acceptable v1 if this proves complex.)
- All three intervals are per-step overridable (step 10 likely needs longer
  defaults, e.g. 30/60/120s, since it's inherently fiddly even when done right).
- These numbers are starting defaults, expected to be retuned after watching
  real trainees — don't treat as fixed.

---

## 7. Timing parameters

```yaml
inference_interval_ms: 200          # ~5 checks/sec — don't run detection every camera frame
stability_window_frames: 10         # ~2 sec of history at above rate
stability_pass_ratio: 0.7           # 7 of last 10 *counted* (non-uncertain) frames must agree
confidence_threshold: 0.60          # global default
```

- `inference_interval_ms`, `stability_window_frames`, `stability_pass_ratio`,
  and `confidence_threshold` must all be **per-step overridable**, falling back
  to these globals when unset. (Lesson from prior system: a single global
  confidence threshold causes "whiplash" tuning — fixing one step's false
  positives breaks another step that was already borderline.)
- Camera capture at native max resolution (e.g. 1920×1080 or whatever the
  webcam supports).
- Run detection at the **highest resolution the model/hardware can sustain**
  at the target inference rate — start high (960/1280 for YOLO, not default
  640), step down only if hardware can't sustain the rate.
- For ROI crops (step 10 especially): explicitly check the **crop's actual
  pixel dimensions** once ROI is defined, not just source frame resolution — a
  1280px frame can still yield a tiny, low-detail crop of a small connector.
- Keep resize/preprocessing constants in **one shared config value**, used
  identically at training time and live inference — do not let these drift
  into two hardcoded literals in two files.

---

## 8. Config schema (single source of truth)

Everything below lives in one config file (YAML), not hardcoded in pipeline
code:

```yaml
app:
  confidence_threshold: 0.60
  inference_interval_ms: 200
  stability_window_frames: 10
  stability_pass_ratio: 0.7

camera:
  device_index: 0
  width: 1920
  height: 1080
  auto_exposure: true

escalation:
  tier_1_reference_image_after_seconds: 20
  tier_2_reference_video_after_seconds: 45
  tier_3_trainer_alert_after_seconds: 90

steps:
  - id: 1
    name: "Attach 2 short pegs to 7x14 block"
    requires: {block_7x14: 1, short_peg: 2}
    reference_image: assets/reference/step_01_correct.jpg
    reference_video: assets/reference/step_01_correct.mp4
    # per-step overrides optional, fall back to app.* defaults:
    # confidence_threshold: 0.65
    # stability_window_frames: 8
    # escalation: {tier_1_reference_image_after_seconds: 15, ...}

  - id: 2
    name: "Add 7x7 block"
    requires: {block_7x14: 1, short_peg: 2, block_7x7: 1}
    reference_image: assets/reference/step_02_correct.jpg
    reference_video: assets/reference/step_02_correct.mp4

  # ... steps 3-9 follow the same pattern, cumulative requirements per §4 ...

  - id: 10
    name: "Connect motor wires to PeeCee"
    type: sequential_roi              # special-cased, see §5
    roi: peecee_connector_zone
    confidence_threshold: 0.75        # stricter — highest-risk class
    stability_window_frames: 15       # longer — avoid premature confirm
    escalation: {tier_1_reference_image_after_seconds: 30,
                 tier_2_reference_video_after_seconds: 60,
                 tier_3_trainer_alert_after_seconds: 120}
```

---

## 9. Model & training details

- Detector: **YOLOv8s** (start here; YOLO-NAS-s worth an A/B test in parallel
  if bandwidth allows — cheap swap, same pipeline).
- Tracker: **ByteTrack** (built into `ultralytics`) for stable counting /
  occlusion tolerance — not for instance re-identification.
- Loss: standard detection loss, but ensure **class-weighted / balanced
  sampling** across parts — some classes (e.g. `short_peg`, appears in 8 of 10
  steps) will naturally get far more captured frames than others (e.g.
  `castor_wheel`, appears once). Do not let this go unaddressed — it's the
  single most common silent-failure cause in small-class-count detection
  systems (a majority class can dominate the decision boundary even when a
  minority class is visually easy).
- Resolution: see §7 — go as high as hardware sustains at the target inference
  rate, don't default to 640 without checking.
- Post-training: **prompt a reminder to re-export/re-deploy the model** before
  live testing — a common failure mode is testing against a stale model that
  wasn't refreshed after a retrain (no need to build an automated mtime check
  for v1, a simple post-training reminder message is enough).

---

## 10. Labeling plan

- Reuse/relabel any existing footage rather than recollecting from scratch.
- Consider an open-vocab detector (e.g. Grounding DINO) for a first-pass
  auto-label to speed up bounding box creation, then hand-correct.
- Label diversity checklist per class: multiple workers, lighting conditions,
  angles, occlusion states (hand partially covering part), explicit hard
  negatives (similar-looking wrong part, empty zone, mid-motion blur).
- Extra labeling care on the high-risk pairs from §3, and especially on
  `motor_lead_loose` / `motor_lead_connected` within the step 10 ROI crop.
- Split train/val **by worker/session**, not randomly — validation must
  include at least one session unseen in training.

---

## 11. Build order

1. Config schema (§8) + loader/validation — everything else reads it.
2. Camera capture wrapper (threaded, always-latest-frame buffer).
3. Data capture tooling — unblocks labeling, the actual long pole.
4. YOLO training pipeline (class-weighted sampling from day one).
5. ONNX/deployment export step, with the post-training export reminder (§9).
6. Detection + ByteTrack integration.
7. Stability window logic (3-way outcome, per-step overridable window/ratio/
   threshold).
8. Rule engine / step state machine (linear sequence, cumulative requirements,
   step 10's special sequential-ROI sub-flow).
9. Escalation timer logic (tiered image → video → trainer alert, per-step
   overridable intervals, timer reset behavior).
10. Feedback/UI layer (current step, live progress, tier escalation display,
    step 10's "(1/2)/(2/2)" sub-progress).
11. Headless smoke test of the full pipeline before building final UI polish.
12. Trainer-facing alert/dashboard view (even minimal — a flagged-steps list
    is enough for v1).

---

## 12. Explicitly deferred (not v1, revisit if needed)

- Confusion matrix / self-check / thumbnail-audit diagnostic scripts — skip
  for now, add only if something looks broken during testing.
- Automated stale-export mtime checking — a simple post-training reminder
  message is sufficient for v1.
- DAG-based step dependencies / flexible groups — steps are strictly linear
  right now; only revisit if a non-linear step (like the earlier flexible
  wheel-axle idea) gets reintroduced.
