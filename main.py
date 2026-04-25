from __future__ import annotations

import argparse
import json
from pathlib import Path

from vectorizer.config import load_config
from vectorizer.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="位图到参数化矢量化")
    parser.add_argument("--image", required=True, help="输入图像路径")
    parser.add_argument("--config", default="config/default.yaml", help="配置文件路径")
    parser.add_argument("--output", default="data/output", help="输出根目录")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    result = run_pipeline(args.image, cfg, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
