"""Launch the local JunQi replay viewer."""

from __future__ import annotations

import argparse
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path, help="manual or policy .npz replay")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SystemExit(
            'Replay UI dependencies are missing; install with pip install -e ".[viz]"'
        ) from exc

    from .app import create_app

    uvicorn.run(
        create_app(str(args.replay)),
        host=args.host,
        port=args.port,
        log_level="info",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
