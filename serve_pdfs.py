#!/usr/bin/env python3
"""
Minimal static file server (stdlib http.server) that serves the PDF directory.

Default:  http://0.0.0.0:8080/
PDFs appear at: http://localhost:8080/01_aws_rds_aurora.pdf  etc.

This lets any HTTP client (including LangChain loaders that accept URLs,
or a browser) read the PDFs. The actual parse/chunk/embed work is done
by ingest.py.
"""

import argparse
import os
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Serve PDFs with Python http.server")
    parser.add_argument("--dir", default=os.getenv("PDF_DIR", "/data/pdfs"),
                        help="Directory containing PDFs (default: $PDF_DIR or /data/pdfs)")
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PDF_PORT", "8080")))
    args = parser.parse_args()

    root = Path(args.dir).resolve()
    if not root.is_dir():
        raise SystemExit(f"Directory not found: {root}")

    # Change into the directory so SimpleHTTPRequestHandler serves relative paths
    os.chdir(root)

    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"Serving PDFs from {root}")
    print(f"  → http://{args.host}:{args.port}/")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
