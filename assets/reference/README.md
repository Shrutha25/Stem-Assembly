# Reference media

For the full capture → label → train → test pipeline and commands, see the
[project README](../../README.md). This file only covers the media below.

Per-step reference images/videos referenced by `config/default_config.yaml`
(`reference_image` / `reference_video` for each step) live here, e.g.:

```
assets/reference/step_01_correct.jpg
assets/reference/step_01_correct.mp4
assets/reference/step_02_correct.jpg
assets/reference/step_02_correct.mp4
assets/reference/step_03_correct.jpg
assets/reference/step_03_correct.mp4
assets/reference/step_04_correct.jpg
assets/reference/step_04_correct.mp4
assets/reference/step_05_correct.jpg
assets/reference/step_05_correct.mp4
```

None exist yet. The trainee UI degrades gracefully (shows "reference
image/video not found yet") when a configured path is missing, so the rest
of the system works without these — this is a nice-to-have polish item, not
a blocker for training or testing the pipeline.

**Not the same thing as `data/sessions/`.** That's raw training-data capture
(many photos per step, used to train the detector). This folder is the
opposite: one curated "here's what correct looks like" image and a short
video per step, shown to the trainee in the escalation UI (tier 1 = image,
tier 2 = video) when they're stuck — polish for a human to *look at*, not
training data for the model.

To create these: once a step is reliably reproducible, take one clean photo
of it done correctly and a short (10-30s) screen-recorded or phone video of
doing it from scratch, named to match `config/default_config.yaml`'s
`reference_image`/`reference_video` paths for that step (the list above).
