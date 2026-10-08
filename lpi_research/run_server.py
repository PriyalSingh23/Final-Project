"""Launcher script for Project ANSHUMAN LPI-CGAN Transceiver Web Application.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import argparse
import uvicorn

# Ensure repository root is on Python search path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    parser = argparse.ArgumentParser(description="LPI-CGAN EW Transceiver Web Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")
    args = parser.parse_args()

    print("=" * 70)
    print("PROJECT ANSHUMAN -- LPI-CGAN TRANSCEIVER & EW DASHBOARD")
    print(f"Launching web interface on http://{args.host}:{args.port}")
    print("=" * 70)

    uvicorn.run("lpi_research.app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
