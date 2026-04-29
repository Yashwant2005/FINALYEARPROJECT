from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from docx import Document


def append_document(target: Document, source: Document) -> None:
    for element in source.element.body:
        target.element.body.append(deepcopy(element))


def merge_word_files(first_file: Path, second_file: Path, output_file: Path) -> None:
    if not first_file.exists():
        raise FileNotFoundError(f"First file not found: {first_file}")
    if not second_file.exists():
        raise FileNotFoundError(f"Second file not found: {second_file}")
    if first_file.suffix.lower() != ".docx" or second_file.suffix.lower() != ".docx":
        raise ValueError("This script currently supports only .docx files.")

    merged_document = Document(first_file)
    second_document = Document(second_file)

    append_document(merged_document, second_document)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged_document.save(output_file)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge two Word .docx files into one output document."
    )
    parser.add_argument("first_file", type=Path, help="Path to the first .docx file")
    parser.add_argument("second_file", type=Path, help="Path to the second .docx file")
    parser.add_argument("output_file", type=Path, help="Path for the merged .docx file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_word_files(args.first_file, args.second_file, args.output_file)
    print(f"Merged file created: {args.output_file}")


if __name__ == "__main__":
    main()
