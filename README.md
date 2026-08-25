# Assembly Trainer

A continuous-vision training-station app: a fixed camera watches a trainee
build a physical assembly step by step, auto-detects progress (no button
press), and gives tiered escalating help if they're stuck. Full design
rationale is in [ASSEMBLY_TRAINER_BUILD_PLAN.md](ASSEMBLY_TRAINER_BUILD_PLAN.md) —
this file is the practical walkthrough: how to go from an empty `data/`
folder to a trained model you can run live.

## Setup

```
pip install -r requirements.txt
```

Everything tunable (classes, steps, camera, thresholds, escalation timing)
lives in one file: [`config/default_config.yaml`](config/default_config.yaml).
You generally won't need to touch code to change parts/steps — edit that
file.

## The pipeline, start to finish

### 1. Capture training photos

```
python -m assembly_trainer.data_capture --keypress
```

Opens a live camera preview with the current step's name and required parts
shown on screen. Build through the assembly for real while capturing:

- **SPACE** — capture the current frame
- **N** / **P** — move to the next/previous step (the prompt does not
  auto-advance; you're capturing data, not running the trained pipeline)
- **O** — toggle "wrong orientation negative" mode for the current step (if
  it has one — see `config.yaml`'s `step<N>_wrong_orientation` classes).
  Stage that step's part deliberately wrong (flipped/rotated) before
  capturing. Resets to normal automatically on N/P.
- **ESC** — quit

Every run creates its own timestamped session folder under
`data/sessions/`. Do several **separate sittings** (different lighting,
full rebuilds from scratch) — session-level variety is what makes the later
train/val/test split meaningful instead of just memorized near-duplicates.

To add more photos to a session you already started instead of starting a
new one:

```
python -m assembly_trainer.data_capture --keypress --resume-session "<session-folder-name>" --step 3
```

(`--session "<label>"` on a fresh run just tags the new folder with a
human-readable label — it's cosmetic, every run gets a unique folder either
way.)

### 2. Organize into train/val/test

```
python -m assembly_trainer.training.split_sessions
```

Splits every session's frames across train/val/test (not whole-session
holdout — every session and every step ends up represented in every split),
copies them into `data/dataset/images/{train,val,test}/`, and seeds
`data/dataset/data.yaml` with the current class list. Safe to re-run
anytime after capturing more data — it's deterministic and won't reshuffle
frames already assigned.

### 3. Label the images

Bounding boxes per part, per image — this part is manual, no way around it.
See [`data/dataset/labels/README.md`](data/dataset/labels/README.md) for
the exact format, class-id order, and tool recommendations (Roboflow or
labelImg both export directly to what's needed here).

Then check your work:

```
python -m assembly_trainer.training.verify_labels
```

Confirms every image has a matching label file, every class id is valid,
and shows per-class instance counts — catches mistakes before they waste a
training run.

### 4. Train

```
python -m assembly_trainer.training.train --data data/dataset/data.yaml --epochs 100
```

Fine-tunes YOLOv8s with class-balanced oversampling (classes that appear in
fewer steps get less data by default — this compensates). Uses a GPU
automatically if one's available (`torch.cuda.is_available()`), otherwise
CPU. Weights land in `models/runs/assembly_trainer/weights/best.pt`.

### 5. Export

```
python -m assembly_trainer.training.export --weights models/runs/assembly_trainer/weights/best.pt --update-config
```

Exports to ONNX and updates `config/default_config.yaml`'s `model_path` to
point at it. **Re-run this after every retrain** — testing against a stale
model is the most common "why isn't it detecting anything new" mistake.

### 6. Run it live

```
python -m assembly_trainer.app.server
```

Then open `http://127.0.0.1:8000` (trainee view) or `/trainer` (trainer
dashboard, flagged-steps list). Uses the real camera and the trained model
from `config.yaml`'s `model_path`.

Useful variants:

```
python -m assembly_trainer.app.server --mock                 # scripted fake detections, no trained model needed
python -m assembly_trainer.app.server --mock --no-camera     # no hardware at all, pure UI/plumbing check
python -m assembly_trainer.app.server --device cpu           # force CPU even if a GPU is present
```

### 7. Headless smoke test (no camera, no model, no network)

```
python scripts/smoke_test.py
```

Pure-logic test of the state machine, stability window, escalation timer,
and diagnosis logic using scripted synthetic detections. Useful for
verifying the pipeline itself still behaves correctly after any code change
— run this any time something seems off before suspecting your data/model.

## Project layout

```
config/default_config.yaml     single source of truth for classes, steps, timing, thresholds
assembly_trainer/
  data_capture.py               capture tool (step 1 above)
  training/split_sessions.py    train/val/test split (step 2)
  training/verify_labels.py     label sanity check (step 3)
  training/train.py             training (step 4)
  training/export.py            ONNX export (step 5)
  app/server.py                 live web UI (step 6)
  config.py, state_machine.py, gating.py, stability.py, escalation.py, roi.py
                                 the actual detection/rule-engine pipeline
data/sessions/                  raw captured photos + manifest.jsonl per session
data/dataset/                   organized images/labels for training (generated by split_sessions.py)
assets/reference/                per-step "here's what correct looks like" media for the trainee UI (optional polish)
scripts/smoke_test.py           headless pipeline test (step 7)
```

## Troubleshooting

- **Camera hangs on open**: on Windows, `cv2.VideoCapture`'s default
  backend can hang indefinitely if it can't get access (another app has it
  open, or a driver quirk) instead of failing fast. `camera.py` already
  forces DirectShow (`cv2.CAP_DSHOW`) to avoid this — if it still hangs,
  something else has the camera (check Settings → Privacy → Camera →
  Recent activity, or close apps like Teams/WhatsApp/the Windows Camera app).
- **First few captured frames look dim or blown out**: auto-exposure needs
  ~2-2.5s to settle after opening — `camera.warmup_seconds` in the config
  already accounts for this, but if you're calling the camera directly in a
  script, give it a moment before grabbing a frame.
- **`--mock` runs but the state machine never advances past step 1**: expected
  when using a placeholder/pretrained model (e.g. a stock COCO model) instead
  of your actual trained one — it doesn't know your custom classes. Only
  your own exported model (step 5 above) will actually detect the real parts.
