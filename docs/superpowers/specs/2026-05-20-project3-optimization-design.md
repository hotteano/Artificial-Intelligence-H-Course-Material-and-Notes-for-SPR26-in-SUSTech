# Project3 Optimization Design (CPU-only)

## Context
Project3 compares two KGRS implementations. We can modify:
- Project3/reference_submit/kgrs.py
- Project3/compare.py

We must not change:
- Project3/demo/kgrs.py
- Project3/eval/evaluate.py

## Goals
- Improve training runtime on CPU-only while keeping AUC and nDCG stable.
- Improve overall evaluation runtime (CTR and TopK) without changing outputs.
- Improve code clarity and maintainability in the allowed files.

## Non-Goals
- Changing model architecture beyond safe performance tweaks.
- Changing dataset splits or evaluation definitions.
- GPU-specific optimizations.

## Constraints
- CPU-only execution.
- demo and evaluate are read-only.
- Avoid behavior regressions for CTR and TopK outputs.

## Proposed Approach
### 1) Training speed
- Cache negative-sampling index structures and reuse across epochs.
- Refresh negative samples on a clear cadence (every N epochs) to balance speed and diversity.
- Reduce per-batch tensor construction overhead by batching and using `torch.no_grad()` where appropriate.

### 2) Evaluation speed
- CTR evaluation: pre-split batches, reuse relation vectors, and use `torch.no_grad()`.
- TopK evaluation: precompute item list and avoid repeated Python loops where possible; reuse cached structures.

### 3) Code clarity
- Separate negative sampling into `_build_index()` and `_sample_negatives()` helpers.
- Consolidate scoring paths into `score_batch()` and `score_user_items()` helpers in the model.
- In compare.py, reduce repeated I/O and redundant data manipulations.

## Detailed Changes
### reference_submit/kgrs.py
- Dataloader
  - Add cached index structures for (head, rel) and (rel, tail) to speed sampling.
  - Add explicit `refresh` control for negative sampling.
- Model
  - Add `score_batch()` that accepts numpy arrays and returns scores.
  - Add `score_user_items()` to support TopK evaluation with shared setup.
  - Wrap evaluation in `torch.no_grad()`.

### compare.py
- Load KG lines once and pass to both models.
- Avoid repeated concat/shuffle logic by reusing prepared arrays.

## Data Flow
- Load data (per-user 8:1:1 split).
- Initialize model and cached sampling state.
- Training loop with periodic negative-sample refresh.
- CTR evaluation in batches.
- TopK evaluation using cached item list and trained embeddings.

## Testing Plan
- Add minimal unit tests:
  - Negative sampling never returns known positives.
  - `eval_ctr` output shape and deterministic length.
  - `eval_topk` returns k items per user and excludes known positives.
- Run compare.py once to validate metrics and timing logs.

## Risks and Mitigations
- Risk: stale negative samples reduce quality.
  - Mitigation: configurable refresh cadence and default to every few epochs.
- Risk: refactor introduces subtle behavior changes.
  - Mitigation: tests and side-by-side metric comparison.

## Rollout
- Apply refactors and tests in a single change set.
- Validate on current data and compare AUC/nDCG to baseline.
