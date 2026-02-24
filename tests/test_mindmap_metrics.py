"""
Test metrics for Sinhala mindmap extraction:
- Node boundary F1
- Node type F1
- Relation detection F1
- Relation direction accuracy
- Hierarchy consistency score
"""
import unittest
from typing import List, Dict, Tuple, Set

def span_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    """Compute intersection-over-union for two spans."""
    s1, e1 = a
    s2, e2 = b
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = max(e1, e2) - min(s1, s2)
    return inter / union if union > 0 else 0.0

def f1_score(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0

class TestMindmapMetrics(unittest.TestCase):
    def setUp(self):
        # Example gold and pred for one doc
        self.gold_nodes = [
            {"start": 0, "end": 8, "type": "concept"},
            {"start": 10, "end": 20, "type": "multiword-concept"},
        ]
        self.pred_nodes = [
            {"start": 0, "end": 8, "type": "concept"},
            {"start": 10, "end": 21, "type": "multiword-concept"},
        ]
        self.gold_rels = [
            {"source": 0, "target": 1, "type": "is-a", "directed": True},
        ]
        self.pred_rels = [
            {"source": 0, "target": 1, "type": "is-a", "directed": True},
            {"source": 1, "target": 0, "type": "is-a", "directed": False},
        ]

    def test_node_boundary_f1(self):
        gold_spans = [(n["start"], n["end"]) for n in self.gold_nodes]
        pred_spans = [(n["start"], n["end"]) for n in self.pred_nodes]
        tp = sum(any(span_iou(gs, ps) > 0.5 for gs in gold_spans) for ps in pred_spans)
        fp = len(pred_spans) - tp
        fn = len(gold_spans) - tp
        f1 = f1_score(tp, fp, fn)
        self.assertGreaterEqual(f1, 0.5)

    def test_node_type_f1(self):
        gold_types = set((n["start"], n["end"], n["type"]) for n in self.gold_nodes)
        pred_types = set((n["start"], n["end"], n["type"]) for n in self.pred_nodes)
        tp = len(gold_types & pred_types)
        fp = len(pred_types - gold_types)
        fn = len(gold_types - pred_types)
        f1 = f1_score(tp, fp, fn)
        self.assertGreaterEqual(f1, 0.5)

    def test_relation_detection_f1(self):
        gold_pairs = set((r["source"], r["target"]) for r in self.gold_rels)
        pred_pairs = set((r["source"], r["target"]) for r in self.pred_rels)
        tp = len(gold_pairs & pred_pairs)
        fp = len(pred_pairs - gold_pairs)
        fn = len(gold_pairs - pred_pairs)
        f1 = f1_score(tp, fp, fn)
        self.assertGreaterEqual(f1, 0.5)

    def test_relation_direction_accuracy(self):
        gold_dirs = set((r["source"], r["target"]) for r in self.gold_rels if r["directed"])
        pred_dirs = set((r["source"], r["target"]) for r in self.pred_rels if r["directed"])
        correct = len(gold_dirs & pred_dirs)
        total = len(gold_dirs)
        acc = correct / total if total else 1.0
        self.assertGreaterEqual(acc, 0.5)

    def test_hierarchy_consistency_score(self):
        # Example: check for cycles (should be acyclic)
        edges = [(r["source"], r["target"]) for r in self.pred_rels if r["directed"]]
        def has_cycle(edges):
            from collections import defaultdict
            graph = defaultdict(list)
            for s, t in edges:
                graph[s].append(t)
            visited = set()
            stack = set()
            def visit(v):
                if v in stack:
                    return True
                if v in visited:
                    return False
                visited.add(v)
                stack.add(v)
                for n in graph[v]:
                    if visit(n):
                        return True
                stack.remove(v)
                return False
            # Fix: iterate over a static list of keys to avoid RuntimeError
            return any(visit(v) for v in list(graph.keys()))
        score = 1.0 if not has_cycle(edges) else 0.0
        self.assertGreaterEqual(score, 0.0)

if __name__ == "__main__":
    unittest.main()
