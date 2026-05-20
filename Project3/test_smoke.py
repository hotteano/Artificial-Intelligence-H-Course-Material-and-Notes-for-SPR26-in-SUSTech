import importlib.util
import os
import sys
import unittest

import numpy as np


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_kgrs_class(path: str):
    """Dynamic import to avoid class name collision between demo and reference_submit."""
    module_name = f"kgrs_smoke_{os.path.basename(os.path.dirname(path))}"
    spec = importlib.util.spec_from_file_location(module_name, os.path.abspath(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.KGRS


def make_toy_data():
    """Return tiny synthetic datasets for smoke testing."""
    # 2 users, 3 items, a handful of KG triples
    train_pos = np.array([[0, 1, 1], [1, 2, 1]], dtype=np.int32)
    train_neg = np.array([[0, 2, 0], [1, 0, 0]], dtype=np.int32)
    kg_lines = [
        "0\tfilm.film.star\t1\n",
        "1\tfilm.film.genre\t2\n",
    ]
    return train_pos, train_neg, kg_lines


class KGRSSmokeMixin:
    """Mixin holding the actual smoke-test assertions."""

    kgrs_path: str = ""

    def test_smoke(self):
        original_cwd = os.getcwd()
        KGRS = load_kgrs_class(self.kgrs_path)
        train_pos, train_neg, kg_lines = make_toy_data()

        try:
            kgrs = KGRS(train_pos.copy(), train_neg.copy(), kg_lines)
            # Reduce epochs so the smoke test finishes in a few seconds
            kgrs.epoch_num = 1
            kgrs.training()

            # ---- eval_ctr ----
            test_data = np.array([[0, 1], [1, 0]], dtype=np.int32)
            # shrink eval batch size so tiny test_data does not trigger
            # a division-by-zero in np.array_split inside eval_ctr
            kgrs.eval_batch_size = max(1, len(test_data))
            scores = kgrs.eval_ctr(test_data)
            self.assertIsInstance(scores, np.ndarray)
            self.assertEqual(scores.shape, (len(test_data),))

            # ---- eval_topk ----
            users = [0, 1]
            k = 2
            topk = kgrs.eval_topk(users, k=k)

            self.assertIsInstance(topk, list)
            self.assertEqual(len(topk), len(users))

            # Build map of original user -> known positive items
            known_pos = {}
            for record in train_pos:
                u, i = int(record[0]), int(record[1])
                known_pos.setdefault(u, set()).add(i)

            for user, items in zip(users, topk):
                self.assertIsInstance(items, list)
                self.assertEqual(len(items), k,
                                 f"User {user} should receive exactly {k} recommendations")
                # Recommendations must not contain items already seen in train_pos
                self.assertTrue(
                    known_pos[user].isdisjoint(items),
                    f"Recommended items for user {user} contain a known positive item"
                )
        finally:
            os.chdir(original_cwd)


class TestDemoKGRS(KGRSSmokeMixin, unittest.TestCase):
    kgrs_path = os.path.join(PROJECT_ROOT, "demo", "kgrs.py")


class TestReferenceKGRS(KGRSSmokeMixin, unittest.TestCase):
    kgrs_path = os.path.join(PROJECT_ROOT, "reference_submit", "kgrs.py")


if __name__ == "__main__":
    unittest.main()
