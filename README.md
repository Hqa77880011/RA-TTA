# RA-TTA

Implementation of **RA-TTA: Reliability- and Ambiguity-Aware Adaptation for Vision-Language Tracking**.

The tracker fuses visual, language, and a bounded quality-weighted memory. Visual confidence combines entropy, peak contrast, and memory similarity; language ambiguity compares positive and negative candidate similarity. Only reliable, unambiguous, non-occluded frames can update the local adapter through augmentation consistency and anchoring. Adapter state resets for each input sequence; ground-truth boxes are used only by the source-training objective and evaluator.

## Install

Use Python 3.10 or later. Create and activate a virtual environment, then install:

```bash
python -m pip install -e .
```

## Prepare data

`x[N,T,3,H,W]` contains videos, `initial_box[N,4]` the first-frame xyxy box normalized to [0,1], `language[N,D]` a fixed language embedding, and `y[N,T,4]` reference boxes. Export sequences from LaSOT/OTB language or another licensed tracking dataset; keep whole videos together. Language vectors are supplied by a frozen text encoder with the configured width.

Arrays use numeric NPZ format, loaded without pickle. `group[N]` identifies independent subjects, devices, scenes, lots, or source images. Related samples must share a group. Optional `split[N]` codes 0/1/2/3 explicitly assign training/validation/calibration/test (three-role input using 0/1/2 remains supported); otherwise ordered groups are allocated 60/15/10/15 to training/validation/calibration/test. Test data never select model parameters or checkpoints.

```bash
ra-tta demo --output data/example.npz --count 96 --config config.json
ra-tta prepare --input data/example.npz --output data/prepared
```

The demo defines a small synthetic problem with the documented tensor interface. Use application data in that schema for the paper's evaluation settings.

## Train and execute

```bash
ra-tta train --input data/prepared --output runs/full --config config.json --variant full --epochs 100 --batch-size 16
ra-tta evaluate --input data/prepared/test.npz --model runs/full/model.pt --output runs/full/evaluation
ra-tta plot --input runs --output runs/figures
```

Training uses AdamW and selects the minimum validation objective. `model.pt` stores parameters and configuration; `history.json` records training/validation objectives. Evaluation exports predictions and mechanism tensors to `predictions.npz` and computed scores to `metrics.json`. Plotting exports comparison charts from those actual evaluation files.

`--device cpu` is the default; `--device cuda` selects CUDA. `--lr`, `--epochs`, `--batch-size`, and `--seed` control optimization. Method parameters are in `config.json`; `--set key=value` overrides one value using JSON syntax. Use identical splits and seeds across comparisons.

## Calibration

For interval or classification confidence calibration, use independent groups excluded from fitting and checkpoint selection:

```bash
ra-tta calibrate --input data/prepared/calibration.npz --model runs/full/model.pt --output runs/full/calibrated.pt
```

Evaluate `calibrated.pt` with the same evaluation command. Classification uses temperature scaling; numeric intervals use a 90% group-maximum split-calibration expansion. Insufficient independent units retain infinite intervals. Pretraining checkpoints can initialize supervised training with `train --init path/to/model.pt`.

## Baselines and ablations

Available variants: `full`, `baseline`, `no_gate`, `no_language`, `no_memory`. `baseline` removes the paper-specific mechanism while retaining the input and prediction task. Other variants disable their named component.

```bash
ra-tta train --input data/prepared --output runs/baseline --config config.json --variant baseline
ra-tta evaluate --input data/prepared/test.npz --model runs/baseline/model.pt --output runs/baseline/evaluation
```

## Metrics and implementation conventions

Classification reports accuracy, balanced accuracy, macro-F1, NLL, confidence ECE, and worst-group F1; binary tasks add AUROC and AUPRC. Unknown labels use -1 and enable unknown AUROC and OSCR. Segmentation reports Dice, IoU, precision, recall, MAE, Brier/ECE, two-pixel boundary F1, symmetric contour distance, and HD95. Reconstruction reports RMSE, MAE, signed bias, unit-range PSNR, and Gaussian-window SSIM. Each method adds its mechanism metrics where applicable.

Parameter counts and forward timing describe the executing process. Use batch size one and a declared device configuration for device latency comparisons. Physical energy and scores from external task evaluators require measured inputs; they are not inferred from parameter counts. All prediction arrays remain available for task-specific evaluation.

The implementation uses the explicit core operations described above with configurable compact networks. Numeric conventions and application-specific information are stated with the method and data schema. The parameter file contains the current defaults.

## Reproduction logic conventions

The no_language control sets the language fusion weight to zero and removes its ambiguity requirement from the update gate. Its predicted boxes are independent of the supplied language embedding; visual reliability and occlusion still gate updates.

## Selected compact tracking setting

`config.json` fixes width 24, a four-frame 16-by-16 RGB training episode,
adapt_lr=0.01 and eight memory entries. Use an explicit 24-dimensional frozen
auxiliary language embedding as the input interface; fitting a VLM is unnecessary
for this setting. The self-contained control generates moving red/green/blue
target rectangles and a differently colored distractor; its fixed color-word code
is the repeated RGB one-hot vector (24 coordinates). This is a declared closed
three-word vocabulary, without pretending to use a general text model. Actual
text-conditioned tracking can supply independently frozen vectors of this dimension. The adapter is reset per sequence, and no_language is
the input-invariant control.

```bash
ra-tta demo --output data/example.npz --config config.json --count 96
ra-tta prepare --input data/example.npz --output data/prepared
ra-tta train --input data/prepared --config config.json --output runs/compact --variant full --epochs 100 --seed 13
ra-tta evaluate --input data/prepared/test.npz --model runs/compact/model.pt --output runs/compact/evaluation
```

Real inputs use x[N,T,3,H,W], language[N,24], normalized initial_box[N,4], y[N,T,4]
and independent video group IDs. No benchmark-specific loader is required.
Pixel positions are cell centers (i+0.5)/size, consistent with edge-normalized
ground-truth boxes. Predicted coordinate pairs are ordered before clipping, preserving valid xyxy
boxes even when the learned coordinate offsets cross. Normalized precision is
explicitly the fraction of center errors <=0.05 in unit image coordinates.
