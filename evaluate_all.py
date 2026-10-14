import sys
import os
import io

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
elif sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Ensure root directory in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.model_evaluator import ModelEvaluator


def main():
    evaluator = ModelEvaluator()
    benchmark_res = evaluator.run_benchmark()
    table_str = evaluator.format_table(benchmark_res)
    print(table_str)


if __name__ == "__main__":
    main()
