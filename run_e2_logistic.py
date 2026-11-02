"""E2 方案 A：NSTEMI vs STEMI，唯一分类器为逻辑回归。"""

from __future__ import annotations

import sys

from logistic_regression_classifier import main


if __name__ == "__main__":
    sys.exit(main(fixed_experiment="e2"))
