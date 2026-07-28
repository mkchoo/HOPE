# Training records

This directory preserves the records from the completed EMA curriculum:

1. `K1_wu0`: one-day rollout, no warm-up, 250 epochs.
2. `K4_wu1`: initialized from the K1 EMA checkpoint, 100 epochs.
3. `K8_wu2`: initialized from the K4 EMA checkpoint, 100 epochs.

Each stage contains:

- `args.json`: exact historical arguments recorded by the original, more
  configurable research script. The simplified public script fixes these
  settings instead of exposing them as options.
- `scalars.csv`: all TensorBoard scalar records in long-form CSV.
- `tensorboard/`: the original TensorBoard event and hparams files.

`training_summary.csv` contains the final aggregate metrics. These are weighted
MAEs across variables with different units after denormalization, so the
denormalized aggregate should not be interpreted as a single physical unit.

View the original event files with:

```bash
tensorboard --logdir logs
```
