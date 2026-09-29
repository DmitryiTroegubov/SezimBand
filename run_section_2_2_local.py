"""Execute the supplied Section 2.2 notebook against the local UrbanSound8K data.

The notebook's model, preprocessing, and training cells are left intact. This
runner only replaces Kaggle paths and resolves source-recording overlap between
the predefined train/validation/test partitions before feature extraction.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import textwrap
import time

import nbformat


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"Expected one occurrence of {old!r}, got {source.count(old)}")
    return source.replace(old, new)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--notebook",
        type=Path,
        default=Path(r"C:\Users\dimat\Downloads\SezimBand_Section_2_2.ipynb"),
    )
    parser.add_argument("--data", type=Path, default=Path("UrbanSound8K"))
    parser.add_argument("--out", type=Path, default=Path("sezimband_2_2"))
    args = parser.parse_args()

    metadata = (args.data / "metadata" / "UrbanSound8K.csv").resolve()
    if not metadata.is_file():
        raise FileNotFoundError(metadata)
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    notebook = nbformat.read(args.notebook, as_version=4)
    config_cell = notebook.cells[2]
    config_cell.source = replace_once(
        config_cell.source, "METADATA_CSV = None", f"METADATA_CSV = {str(metadata)!r}"
    )
    config_cell.source = replace_once(
        config_cell.source,
        "SEARCH_ROOT = Path('/kaggle/input')",
        f"SEARCH_ROOT = Path({str(metadata.parent)!r})",
    )
    config_cell.source = replace_once(
        config_cell.source,
        "OUT = Path('/kaggle/working/sezimband_2_2') if Path('/kaggle/working').exists() else Path.cwd() / 'sezimband_2_2'",
        f"OUT = Path({str(output)!r})",
    )

    overlap_resolution = textwrap.dedent("""
        original_rows = len(df)
        _partition = np.select(
            [df.fold.isin(TRAIN_FOLDS), df.fold.isin(VAL_FOLDS), df.fold.isin(TEST_FOLDS)],
            ['train', 'validation', 'test'], default='unknown')
        _rank = pd.Series(_partition, index=df.index).map({'train': 0, 'validation': 1, 'test': 2})
        if _rank.isna().any():
            raise ValueError('Unexpected fold in UrbanSound8K metadata')
        _highest_rank = _rank.groupby(df.fsID).transform('max')
        excluded_overlap = df.loc[_rank < _highest_rank].copy()
        excluded_overlap['original_partition'] = _partition[_rank < _highest_rank]
        excluded_overlap.to_csv(OUT / 'excluded_source_overlap.csv', index=False)
        df = df.loc[_rank == _highest_rank].reset_index(drop=True)
        save_json('source_overlap_resolution.json', {
            'original_excerpt_count': int(original_rows),
            'retained_excerpt_count': int(len(df)),
            'excluded_excerpt_count': int(len(excluded_overlap)),
            'excluded_fsIDs': sorted(int(x) for x in excluded_overlap.fsID.unique()),
            'rule': 'For fsID crossing partitions, retain only the highest-priority partition: test > validation > train.',
            'details_csv': 'excluded_source_overlap.csv',
        })
        print(f'Excluded {len(excluded_overlap)} overlapping source excerpts from {original_rows} originals')
    """).lstrip()
    notebook.cells[4].source = replace_once(
        notebook.cells[4].source,
        "df = pd.read_csv(metadata_path).reset_index(drop=True)\n",
        "df = pd.read_csv(metadata_path).reset_index(drop=True)\n" + overlap_resolution,
    )
    notebook.cells[19].source = replace_once(
        notebook.cells[19].source,
        "The classifier was developed using UrbanSound8K. The dataset used in this run contained",
        "The classifier was developed using UrbanSound8K. The original metadata contained {original_rows} excerpts; {len(excluded_overlap)} training excerpts whose source recording also appeared in validation were excluded to prevent leakage. The dataset used in this run contained",
    )
    notebook.cells[0].source += (
        "\nLocal run note: this copy uses local dataset/output paths and excludes source-"
        "recording overlap before preprocessing; see source_overlap_resolution.json.\n"
    )
    adapted_path = output.parent / "SezimBand_Section_2_2_local.ipynb"
    nbformat.write(notebook, adapted_path)
    print(f"Adapted notebook: {adapted_path}", flush=True)

    namespace = {"__name__": "__main__", "__file__": str(adapted_path)}
    started = time.perf_counter()
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        print(f"\n=== Running notebook cell {index} ===", flush=True)
        exec(compile(cell.source, f"{adapted_path}:cell_{index}", "exec"), namespace)
        print(f"=== Finished cell {index}; elapsed {time.perf_counter() - started:.1f}s ===", flush=True)
    print(f"Notebook run completed in {time.perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
