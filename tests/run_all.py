"""Run the whole QAC test suite with plain python (no pytest):  python qac/tests/run_all.py"""
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODULES = ["tests.test_quantum_math", "tests.test_baselines", "tests.test_metrics",
           "tests.test_affect"]

if __name__ == "__main__":
    total = 0
    for mname in MODULES:
        mod = importlib.import_module(mname)
        fns = [v for k, v in sorted(vars(mod).items()) if k.startswith("test_")]
        for fn in fns:
            fn()
            total += 1
            print(f"  PASS  {mname.split('.')[-1]}.{fn.__name__}")
    print(f"\nAll {total} tests passed.")
