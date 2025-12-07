"""Simple performance benchmark for SinhalaMindMapGenerator.generate

Run with:
    python perf/run_perf.py

Produces basic timing statistics for small/medium/long inputs.
"""
import time
import statistics
import json
import os
import sys

# Ensure project root is on sys.path so imports work when running this script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mindmap_generator import SinhalaMindMapGenerator


def time_it(func, iterations=10):
    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        func()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)  # ms
    return times


def stats(times):
    return {
        'count': len(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'avg_ms': statistics.mean(times),
        'median_ms': statistics.median(times),
        'p95_ms': statistics.quantiles(times, n=20)[18] if len(times) >= 20 else max(times)
    }


def make_text(words=50, sentences=5):
    # Build synthetic Sinhala-like text by repeating sample phrases
    sample = "ශ්‍රී ලංකාව සුන්දර දිවයිනකි"
    sent = (sample + "。") * (words // 5)
    return " ".join([sent for _ in range(sentences)])


def bench():
    gen = SinhalaMindMapGenerator()

    inputs = {
        'small': make_text(words=20, sentences=2),
        'medium': make_text(words=80, sentences=6),
        'long': make_text(words=300, sentences=20)
    }

    results = {}
    for name, txt in inputs.items():
        # choose iterations based on size
        iters = 120 if name == 'small' else 60 if name == 'medium' else 20

        def run():
            gen.generate(txt)

        times = time_it(run, iterations=iters)
        results[name] = stats(times)

    print(json.dumps({'timestamp': time.time(), 'results': results}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    bench()
