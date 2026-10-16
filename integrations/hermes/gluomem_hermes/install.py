"""Copy this dependency-free directory provider into an explicitly chosen profile.

Does not activate it, change config, or install into Hermes' managed environment.
"""

import argparse
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hermes-home", required=True, help="Explicit target profile home; no implicit default"
    )
    args = parser.parse_args()
    target = Path(args.hermes_home).expanduser().resolve() / "plugins" / "gluomem"
    if target.exists():
        parser.error(
            "target already exists; back it up and remove it before upgrading: " + str(target)
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        Path(__file__).parent, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    print("Installed directory provider at " + str(target))
    print("Not enabled. Run hermes memory setup in that profile when ready.")


if __name__ == "__main__":
    main()
