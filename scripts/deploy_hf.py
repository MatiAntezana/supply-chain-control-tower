"""Deploy the demo to Hugging Face Spaces (Docker SDK, free tier).

Requires a logged-in HF account:  hf auth login   (write token)
Usage:  conda run -n AIEnv python scripts/deploy_hf.py [space_name]
"""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi, whoami

REPO_ROOT = Path(__file__).resolve().parents[1]

SPACE_README = """---
title: Supply Chain Control Tower
emoji: "📦"
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---
{body}
"""


def main() -> None:
    space_name = sys.argv[1] if len(sys.argv) > 1 else "supply-chain-control-tower"
    user = whoami()["name"]
    repo_id = f"{user}/{space_name}"
    api = HfApi()
    api.create_repo(repo_id, repo_type="space", space_sdk="docker", exist_ok=True)

    # HF Spaces needs the YAML header on top of the README.
    body = (REPO_ROOT / "README.md").read_text()
    (REPO_ROOT / "README.hf.md").write_text(SPACE_README.format(body=body))

    api.upload_folder(
        repo_id=repo_id, repo_type="space", folder_path=str(REPO_ROOT),
        ignore_patterns=[".git*", "mlruns*", "mlflow.db", "__pycache__",
                         ".pytest_cache", ".ruff_cache", "data/raw*",
                         "README.md", "README.hf.md", "docs/pitch.md"],
    )
    api.upload_file(repo_id=repo_id, repo_type="space",
                    path_or_fileobj=str(REPO_ROOT / "README.hf.md"),
                    path_in_repo="README.md")
    print(f"deployed: https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    main()
