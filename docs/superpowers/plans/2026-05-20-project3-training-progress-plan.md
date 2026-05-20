# Project3 Training Progress + Speed Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add visible training progress logs and improve CPU-only training/evaluation speed for Project3 while keeping metrics stable.

**Architecture:** Add a small progress-format helper in compare.py and use it each epoch. Refactor reference_submit/kgrs.py to reuse cached negative-sampling structures, add small scoring helpers, and wrap evaluation in `torch.no_grad()`.

**Tech Stack:** Python 3.10+, NumPy, PyTorch (CPU), unittest.

---

## File Structure
- Modify: `Project3/compare.py` (progress formatting + logging, reuse KG lines)
- Modify: `Project3/reference_submit/kgrs.py` (sampling cache, eval helpers, no_grad)
- Create: `Project3/tests/test_compare_progress.py`
- Create: `Project3/tests/test_reference_kgrs_helpers.py`

---

### Task 1: Add progress formatting helper + tests

**Files:**
- Create: `Project3/tests/test_compare_progress.py`
- Modify: `Project3/compare.py`

- [ ] **Step 1: Write the failing test**

```python
import unittest

from Project3 import compare


class ProgressFormatTests(unittest.TestCase):
    def test_format_progress_line_includes_epoch_and_elapsed(self):
        line = compare.format_progress_line(epoch=3, max_epochs=10, elapsed_s=12.3)
        self.assertIn("Epoch 03/10", line)
        self.assertIn("12.3s", line)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest Project3/tests/test_compare_progress.py -v`
Expected: FAIL with `AttributeError: module 'Project3.compare' has no attribute 'format_progress_line'`

- [ ] **Step 3: Write minimal implementation**

Add to `Project3/compare.py`:

```python
def format_progress_line(epoch: int, max_epochs: int, elapsed_s: float) -> str:
    return f"  Epoch {epoch:02d}/{max_epochs:02d} | elapsed {elapsed_s:.1f}s"
```

Use it in the training loop (inside `run_single`):

```python
epoch_start = time.time()
for epoch in range(1, max_epochs + 1):
    kgrs.model.train_TransE(epoch_num=1, output_log=False)
    epoch_elapsed = time.time() - epoch_start
    print(format_progress_line(epoch, max_epochs, epoch_elapsed))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest Project3/tests/test_compare_progress.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Project3/compare.py Project3/tests/test_compare_progress.py
git commit -m "Add training progress formatting in compare"
```

---

### Task 2: Add helper tests for reference_submit KGRS

**Files:**
- Create: `Project3/tests/test_reference_kgrs_helpers.py`

- [ ] **Step 1: Write the failing tests**

```python
import os
import unittest
import numpy as np

from Project3.reference_submit.kgrs import Dataloader, RotatE


class KGRSHelperTests(unittest.TestCase):
    def setUp(self):
        module_dir = os.path.join(os.path.dirname(__file__), "..", "reference_submit")
        self.rel_file = os.path.join(module_dir, "relation2id.txt")
        with open(self.rel_file, encoding="utf-8") as f:
            rel_name = f.readline().split("\t")[0]
        self.kg_lines = [f"0\t{rel_name}\t1\n"]
        # user,item,label
        self.train_pos = np.array([[0, 0, 1], [1, 1, 1]], dtype=np.int32)
        self.train_neg = np.array([[0, 1, 0], [1, 0, 0]], dtype=np.int32)

    def test_build_index_contains_known_triples(self):
        dl = Dataloader(self.train_pos, self.train_neg, self.kg_lines, self.rel_file)
        hr_tail_set, rt_head_set = dl._build_index()
        for h, r, t in dl.kg + dl.known_neg_dict:
            self.assertIn(t, hr_tail_set[(h, r)])
            self.assertIn(h, rt_head_set[(r, t)])

    def test_score_batch_matches_forward(self):
        dl = Dataloader(self.train_pos, self.train_neg, self.kg_lines, self.rel_file)
        model = RotatE(ent_num=dl.ent_num, rel_num=dl.rel_num, dataloader=dl, device_index=-1)
        rel_id = dl.rel_dict["feedback_recsys"]
        batch = np.array([[0, 0]], dtype=np.int32)
        scores = model.score_batch(batch, rel_id)
        expected = model.forward([0 + dl.n_entity], [rel_id], [0]).cpu().numpy().squeeze()
        self.assertAlmostEqual(float(scores[0]), float(expected), places=5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m unittest Project3/tests/test_reference_kgrs_helpers.py -v`
Expected: FAIL with `AttributeError: 'Dataloader' object has no attribute '_build_index'` and `AttributeError: 'RotatE' object has no attribute 'score_batch'`

- [ ] **Step 3: Implement minimal helpers in reference_submit/kgrs.py**

Add to `Dataloader`:

```python
    def _build_index(self):
        hr_tail_set = {}
        rt_head_set = {}
        for fact in self.kg + self.known_neg_dict:
            hr_tail_set.setdefault((fact[0], fact[1]), set()).add(fact[2])
            rt_head_set.setdefault((fact[1], fact[2]), set()).add(fact[0])
        return hr_tail_set, rt_head_set
```

Add to `RotatE`:

```python
    def score_batch(self, user_item_pairs: np.ndarray, rel_id: int) -> np.ndarray:
        users = user_item_pairs[:, 0] + self.dataloader.n_entity
        items = user_item_pairs[:, 1]
        rel = [rel_id for _ in range(len(user_item_pairs))]
        with torch.no_grad():
            scores = torch.squeeze(self.forward(users, rel, items), dim=-1)
        return scores.cpu().numpy()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest Project3/tests/test_reference_kgrs_helpers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Project3/reference_submit/kgrs.py Project3/tests/test_reference_kgrs_helpers.py
git commit -m "Add KGRS helper methods for caching and scoring"
```

---

### Task 3: Use cached sampling + no_grad in eval

**Files:**
- Modify: `Project3/reference_submit/kgrs.py`

- [ ] **Step 1: Write the failing test**

Add to `Project3/tests/test_reference_kgrs_helpers.py`:

```python
    def test_cached_negatives_reused_when_refresh_false(self):
        dl = Dataloader(self.train_pos, self.train_neg, self.kg_lines, self.rel_file)
        dl.get_training_batch(refresh=True)
        cached_len = len(dl._cached_neg)
        dl.get_training_batch(refresh=False)
        self.assertEqual(cached_len, len(dl._cached_neg))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest Project3/tests/test_reference_kgrs_helpers.py::KGRSHelperTests.test_cached_negatives_reused_when_refresh_false -v`
Expected: FAIL with `AttributeError: 'Dataloader' object has no attribute '_cached_neg'`

- [ ] **Step 3: Implement minimal caching**

Update `Dataloader.get_training_batch` to set and reuse `_cached_pos` and `_cached_neg`, and use `_build_index` in `_sample_negatives`.

Also wrap CTR/TopK evaluation in `torch.no_grad()` to reduce overhead.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest Project3/tests/test_reference_kgrs_helpers.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Project3/reference_submit/kgrs.py Project3/tests/test_reference_kgrs_helpers.py
git commit -m "Cache negative samples and wrap eval in no_grad"
```

---

### Task 4: Compare-side reuse of KG lines and validation data

**Files:**
- Modify: `Project3/compare.py`

- [ ] **Step 1: Write failing test**

Add to `Project3/tests/test_compare_progress.py`:

```python
    def test_prepare_val_data_returns_labels_and_pairs(self):
        import numpy as np
        val_pos = np.array([[0, 1, 1]], dtype=np.int32)
        val_neg = np.array([[0, 2, 0]], dtype=np.int32)
        val_data, val_labels = compare.prepare_val_data(val_pos, val_neg, seed=1)
        self.assertEqual(val_data.shape, (2, 2))
        self.assertEqual(val_labels.shape, (2,))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest Project3/tests/test_compare_progress.py -v`
Expected: FAIL with `AttributeError: module 'Project3.compare' has no attribute 'prepare_val_data'`

- [ ] **Step 3: Implement minimal helper in compare.py**

Add:

```python
def prepare_val_data(val_pos: np.ndarray, val_neg: np.ndarray, seed: int):
    val_data = np.concatenate((val_neg, val_pos), axis=0)
    rng = np.random.default_rng(seed)
    rng.shuffle(val_data)
    val_labels = val_data[:, 2]
    return val_data[:, :2], val_labels
```

Use in validation to avoid repeat logic and reduce allocations.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m unittest Project3/tests/test_compare_progress.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Project3/compare.py Project3/tests/test_compare_progress.py
git commit -m "Refactor compare validation data prep"
```

---

## Plan Self-Review
- Spec coverage: training speed, eval speed, progress visibility, code clarity covered by Tasks 1-4.
- Placeholder scan: no TODO/TBD.
- Type consistency: helper signatures match usage in tests and code.

---

Plan complete and saved to `docs/superpowers/plans/2026-05-20-project3-training-progress-plan.md`. Two execution options:

1. Subagent-Driven (recommended) - I dispatch a fresh subagent per task, review between tasks, fast iteration
2. Inline Execution - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?