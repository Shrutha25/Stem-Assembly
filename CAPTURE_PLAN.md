# Capture plan — peg detection under current lighting

Why: live pegs read 0.33–0.52 against step 1/2's 0.45 threshold, and pegs at
foreshortened angles aren't detected at all. `peg` is the weakest class
(mAP50 0.907 but mAP50-95 only 0.317 — small, dark, low-texture). Blocks and
the PeeCee are fine (0.98+), so this round is **only about pegs**.

Two gaps to close: **lighting** (training data was warm/neutral, the station
is now blue-cast and dimmer) and **angle** (the block sits rotated, and pegs
at far corners go end-on).

**Before anything, try the free fix:** set `auto_exposure: false` in
`config/default_config.yaml` and restart the server. If the blue cast is the
camera's white balance rather than the room, it may clear on its own and none
of this is needed.

---

## Sessions

Four sittings, each its own session folder. Stop the server first.

```
.venv\Scripts\python.exe -m assembly_trainer.data_capture --keypress --step 1 --session "blue-rot0"
.venv\Scripts\python.exe -m assembly_trainer.data_capture --keypress --step 1 --session "blue-rot45"
.venv\Scripts\python.exe -m assembly_trainer.data_capture --keypress --step 1 --session "blue-rotmixed"
.venv\Scripts\python.exe -m assembly_trainer.data_capture --keypress --step 1 --session "blue-hardangles"
```

Keys: **SPACE** capture · **N/P** next/prev step prompt · **ESC** quit.

Separate sittings matter — sessions are the unit of the train/val/test split,
so variety *across* them is what makes validation meaningful rather than
measuring near-duplicates.

| session | block pose |
|---|---|
| `blue-rot0` | square to the camera, as in the original dataset |
| `blue-rot45` | rotated 45° (the diamond pose that's failing now) |
| `blue-rotmixed` | arbitrary rotations, block moved around the mat |
| `blue-hardangles` | tilted, near/far corners, pegs deliberately end-on |

### Inside every session, cover all three peg counts

This is the part that actually matters — steps 1/2 gate on peg **count**, so
the model must be solid at each rung of the ladder:

- **2 pegs** — bare block, nothing placed (the "not started" state)
- **3 pegs** — one short peg placed (step 1 complete)
- **4 pegs** — both short pegs placed (step 2 complete)

Roughly a third of each session's frames per state. Aim **40–60 frames per
session**, so ~200 frames total → ~600 new peg instances alongside the
existing 1372.

### While capturing

- Vary the block's rotation *within* a session too, not just between them.
- Include frames where a peg is foreshortened or partly occluded — those are
  the ones failing now, and they must be labelled, not avoided.
- Move the block around the mat: near edges, off-centre, closer/further.
- Hands in frame occasionally is good (that's real usage), but most frames
  should be unobstructed.
- Keep the current lighting. Do **not** "fix" the light for capture and then
  deploy under it — capture the condition you'll actually run in.

---

## Merge — READ THIS FIRST

`split_sessions` calls `shutil.rmtree()` on `dataset/images` and rebuilds it
**only** from `data/sessions/`. Your 743 original images have no backing
session on this machine, so a plain run **deletes them**. This has already
happened once; they were recovered from
`C:\Users\sufiy\Downloads\step 1, 2 removed\...`.

Back up first, merge, then restore:

```
robocopy dataset\images dataset\images.backup /E /NFL /NDL /NJH /NJS
robocopy dataset\labels dataset\labels.backup /E /NFL /NDL /NJH /NJS

.venv\Scripts\python.exe -m assembly_trainer.training.split_sessions --dataset-root dataset

robocopy dataset\images.backup dataset\images /E /XC /XN /XO /NFL /NDL /NJH /NJS
robocopy dataset\labels.backup dataset\labels /E /XC /XN /XO /NFL /NDL /NJH /NJS
```

The restore flags `/XC /XN /XO` mean "don't overwrite existing files", so the
newly split frames survive and the originals come back alongside them.
Filenames don't collide: originals are `Desk__*` / `Exp_Centre__*`, new ones
are session-prefixed.

Then confirm nothing was lost — this should show **936 + however many you
captured**, with the new frames as the only unlabelled ones:

```
.venv\Scripts\python.exe -m assembly_trainer.training.verify_labels --dataset-root dataset
```

---

## Label

```
.venv\Scripts\python.exe -m assembly_trainer.label_tool --dataset-root dataset --split train
.venv\Scripts\python.exe -m assembly_trainer.label_tool --dataset-root dataset --split val
.venv\Scripts\python.exe -m assembly_trainer.label_tool --dataset-root dataset --split test
```

- **Label every peg**, including foreshortened and partly occluded ones. Those
  are the failing cases — skipping them teaches the model they're background.
- Both fixture long pegs *and* placed short pegs are the single `peg` class.
  They were merged deliberately: they are not separable (best possible size
  threshold topped out at 78.9%), and steps 1/2 count pegs rather than
  classify them.
- Tight boxes, consistently.
- Label the blocks and PeeCee too if they're in frame — but they're already
  at 0.98+, so don't add sessions just for them.

## Train, export, deploy

```
.venv\Scripts\python.exe -m assembly_trainer.training.train --data dataset/data.yaml ^
    --model yolov8s.pt --epochs 100 --imgsz 960 --device 0 --batch 8

.venv\Scripts\python.exe -m assembly_trainer.training.export --weights models\runs\<run>\weights\best.pt
copy models\runs\<run>\weights\best.onnx models\best.onnx
copy models\runs\<run>\weights\best.pt   models\best.pt
```

`<run>` auto-increments (`assembly_trainer-2`, `-3`, …) — check
`models\runs\` for the newest rather than assuming.

**Never pass `--update-config` to export.** It rewrites the YAML with
`yaml.dump` and destroys every comment in `config/default_config.yaml`. The
config already points `model_path` at `models/best.onnx`, so copying the
weights over is all that's needed.

## Check it worked

Re-measure exact peg-count accuracy on the val set before trusting it live —
the numbers to beat, from the current model:

| conf | all | gt=2 | gt=3 | gt=4 |
|---|---|---|---|---|
| 0.45 | 75% | 63% | 57% | 93% |
| 0.50 | 80% | 72% | 61% | 100% |

`gt=3` (step 1 complete) is the weak rung. If it climbs well above 61%, this
round worked.
