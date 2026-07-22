from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepend the unmodified official filled cover PDF to the compiled Phase 3 paper."
    )
    parser.add_argument("--cover", required=True, help="PDF exported from the official filled cover-page DOCX")
    parser.add_argument("--body", default="writeup/main.pdf")
    parser.add_argument("--output", default="FraQTech_Phase3_Writeup.pdf")
    args = parser.parse_args()
    try:
        from pypdf import PdfReader, PdfWriter
    except ImportError as exc:
        raise RuntimeError("Install pypdf: pip install pypdf") from exc

    cover = Path(args.cover)
    body = Path(args.body)
    if not cover.exists():
        raise FileNotFoundError(cover)
    if not body.exists():
        raise FileNotFoundError(body)
    writer = PdfWriter()
    for source in (cover, body):
        for page in PdfReader(str(source)).pages:
            writer.add_page(page)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        writer.write(f)
    print(f"Created: {output}")


if __name__ == "__main__":
    main()
