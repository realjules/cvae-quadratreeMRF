# Kaggle Session 1: 3-Seed Reproduction Experiments

Copy each cell into a Kaggle notebook cell. Run sequentially.

---

## Cell 1: Setup

```python
!git clone https://github.com/YOUR_USERNAME/cvae-quadratreeMRF.git
%cd cvae-quadratreeMRF
!pip install -q wandb
```

> **NOTE**: Replace the git clone URL with your actual repo. Make sure the latest code is pushed (with `--seed` argument in `complete_training.py`, `extract_pairwise_diagnostics.py`, `plot_paper_figures.py`, and `test_gradient_cosine.py`).

---

## Cell 2: Verify seed argument exists

```python
!grep -n "seed" complete_training.py | head -5
```

You should see `--seed` in the argparse section. If not, the latest code wasn't pushed.

---

## Cell 3: Experiment A — 3 seeds at 10% labels WITH BP

```python
import subprocess

CONTRASTIVE_CKPT = "/kaggle/input/models/udaheju/contrastive-best/pytorch/default/1"

for seed in [0, 1, 2]:
    print(f"\n{'='*60}")
    print(f"  TRAINING: 10% labels, seed={seed}, WITH BP")
    print(f"{'='*60}\n")

    subprocess.run([
        "python", "complete_training.py",
        "--contrastive_ckpt", CONTRASTIVE_CKPT,
        "--output_dir", f"/kaggle/working/output_pct10_seed{seed}",
        "--labeled_percent", "10",
        "--epochs_seg", "30",
        "--seed", str(seed),
        "--data_dir", "./input",
    ], check=True)

    print(f"\n  Seed {seed} complete. Checkpoint: /kaggle/working/output_pct10_seed{seed}/best_segmentation.pth")
```

---

## Cell 4: Experiment F — 3 seeds at 10% labels WITHOUT BP

```python
for seed in [0, 1, 2]:
    print(f"\n{'='*60}")
    print(f"  TRAINING: 10% labels, seed={seed}, NO BP")
    print(f"{'='*60}\n")

    subprocess.run([
        "python", "complete_training.py",
        "--contrastive_ckpt", CONTRASTIVE_CKPT,
        "--output_dir", f"/kaggle/working/output_pct10_nobp_seed{seed}",
        "--labeled_percent", "10",
        "--epochs_seg", "30",
        "--seed", str(seed),
        "--no_bp",
        "--data_dir", "./input",
    ], check=True)

    print(f"\n  Seed {seed} (no-BP) complete.")
```

---

## Cell 5: Cosine similarity measurements (all 3 BP checkpoints)

```python
for seed in [0, 1, 2]:
    print(f"\n{'='*60}")
    print(f"  COSINE SIMILARITY: 10% labels, seed={seed}")
    print(f"{'='*60}\n")

    subprocess.run([
        "python", "test_gradient_cosine.py",
        "--contrastive_ckpt", CONTRASTIVE_CKPT,
        "--seg_ckpt", f"/kaggle/working/output_pct10_seed{seed}/best_segmentation.pth",
    ], check=True)
```

---

## Cell 6: Pairwise diagnostics (all 3 BP checkpoints)

```python
!python extract_pairwise_diagnostics.py \
    --seg_ckpt /kaggle/working/output_pct10_seed0/best_segmentation.pth \
               /kaggle/working/output_pct10_seed1/best_segmentation.pth \
               /kaggle/working/output_pct10_seed2/best_segmentation.pth \
    --data_dir ./input \
    --output_dir /kaggle/working/paper_figures_10pct
```

---

## Cell 7: Generate figures (with whatever data is available)

```python
!python plot_paper_figures.py \
    --output_dir /kaggle/working/paper_figures \
    --pairwise_json /kaggle/working/paper_figures_10pct/pairwise_diagnostics.json
```

---

## Cell 8: Save all outputs (download from Kaggle)

```python
import shutil
import os

# Collect all results into one directory
results_dir = "/kaggle/working/all_results"
os.makedirs(results_dir, exist_ok=True)

# Copy checkpoints
for seed in [0, 1, 2]:
    for prefix in ["output_pct10_seed", "output_pct10_nobp_seed"]:
        src = f"/kaggle/working/{prefix}{seed}/best_segmentation.pth"
        if os.path.exists(src):
            dst = os.path.join(results_dir, f"{prefix}{seed}_best.pth")
            shutil.copy2(src, dst)

# Copy figures and diagnostics
for d in ["paper_figures_10pct", "paper_figures"]:
    src_dir = f"/kaggle/working/{d}"
    if os.path.exists(src_dir):
        shutil.copytree(src_dir, os.path.join(results_dir, d), dirs_exist_ok=True)

# Zip for download
shutil.make_archive("/kaggle/working/session1_results", 'zip', results_dir)
print("Download: /kaggle/working/session1_results.zip")

# List what we got
for root, dirs, files in os.walk(results_dir):
    for f in files:
        path = os.path.join(root, f)
        size = os.path.getsize(path) / 1024
        print(f"  {os.path.relpath(path, results_dir):<60} {size:.1f} KB")
```

---

## Expected runtime

| Cell | What | Time (T4) |
|------|------|-----------|
| 3 | 3x training 10% with BP | ~6h |
| 4 | 3x training 10% no BP | ~6h |
| 5 | 3x cosine measurements | ~15min |
| 6 | Pairwise diagnostics | ~5min |
| 7 | Figure generation | ~1min |
| 8 | Package results | ~1min |
| **Total** | | **~12.5h** |

Fits within Kaggle's 30h/week T4 quota. You'll have ~17h left for Session 2 (50% experiments).

---

## What to check when it finishes

1. **Cell 3 output**: Note the best accuracy for each seed. Are they consistent (within ~5pp)?
2. **Cell 5 output**: Note the Stage 3 cosine values for each seed. Does cos ≈ -0.03 for unary_1.net[0] reproduce?
3. **Cell 6 output**: Check diagonal ratio. Is it consistently low (< 0.3) across seeds?
4. **Cell 8**: Download `session1_results.zip` — it has everything.

---

## Session 2 (run next, separate notebook)

After Session 1 completes, run the same structure but for 50% labels:

```python
# Replace in Cell 3:
#   --labeled_percent 10  →  --labeled_percent 50
#   output_pct10_seed     →  output_pct50_seed

# Skip Cell 4 (no-BP only needed at 10%)
# Run Cells 5-8 with the 50% checkpoints
```

Then populate `plot_paper_figures.py` RESULTS dicts with all 3-seed numbers and regenerate final figures.
