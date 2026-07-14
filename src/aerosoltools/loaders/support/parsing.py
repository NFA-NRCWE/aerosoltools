from __future__ import annotations

import codecs
import os
from collections import Counter
from typing import Any, Callable, List, Sequence, Union

import pandas as pd
from tabulate import tabulate
from tqdm.auto import tqdm

from ...aerosol1d import Aerosol1D
from ...aerosol2d import Aerosol2D
from .exceptions import (
    DelimiterDetectionError,
    EmptyFileError,
    EncodingError,
    LoaderError,
)

__all__ = ["file_list", "load_data_from_folder"]

###############################################################################


def _detect_delimiter(
    file_path: str,
    encodings: list[str] | None = None,
    delimiters: list[str] | None = None,
    sample_lines: int = 10,
    min_count_threshold: int = 3,
    tolerance: int = 1,
) -> tuple[str, str]:
    """Infer file encoding and field delimiter from a text file.

    This helper is used internally by instrument loader functions to robustly
    determine the character encoding and delimiter (comma, semicolon, tab, etc.)
    for CSV-like files.

    The function tries multiple encodings until one succeeds, then inspects a
    sample of non-empty, non-comment lines to find the most consistent delimiter
    based on column counts.

    Args:
        file_path:
            Path to the input text or CSV-like file.
        encodings:
            Encodings to try in order. If ``None``, defaults to
            ``["utf-8", "cp1252", "latin-1"]`` (UTF-8 first, Latin-1 last as the
            can't-fail fallback). A UTF-8/UTF-16/UTF-32 byte-order mark is
            detected up front and always takes precedence.
        delimiters:
            Candidate delimiters to test. If ``None``, defaults to
            ``[",", ";", "\\t", "|"]``.
        sample_lines:
            Number of non-empty lines (from the bottom up) to consider when
            counting delimiter occurrences. Defaults to ``10``.
        min_count_threshold:
            Minimum number of lines that must show consistent delimiter usage
            (within ``tolerance`` of the modal count). Defaults to ``3``.
        tolerance:
            Allowed deviation from the modal delimiter count across sampled
            lines. Defaults to ``1``.

    Returns:
        tuple[str, str]: A pair ``(encoding, delimiter)`` giving the chosen
        encoding and delimiter.

    Raises:
        EncodingError: If none of the candidate encodings can decode the file.
        EmptyFileError: If the file has no usable lines to analyse.
        DelimiterDetectionError: If no reliable delimiter can be inferred.
    """
    # Default trial order: strict UTF-8 first (so genuine UTF-8 files decode
    # correctly), then Windows-1252, then Latin-1 as the guaranteed last resort.
    # NOTE: Latin-1 decodes *any* byte stream without error, so it must come
    # last — historically it was first, which meant UTF-8/UTF-16 were never
    # actually tried and multi-byte UTF-8 files (e.g. "µg/m³") were mis-decoded.
    encodings = encodings or ["utf-8", "cp1252", "latin-1"]
    delimiters = delimiters or [",", ";", "\t", "|"]

    # A byte-order mark is authoritative — honour it ahead of the trial list.
    # UTF-16 is intentionally BOM-only: without a BOM it would happily
    # mis-decode a single-byte (Latin-1) file, so it is never guessed blindly.
    with open(file_path, "rb") as fb:
        head = fb.read(4)
    bom_encoding = None
    if head.startswith(codecs.BOM_UTF8):
        bom_encoding = "utf-8-sig"
    elif head.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        bom_encoding = "utf-16"
    elif head.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        bom_encoding = "utf-32"

    trial = ([bom_encoding] if bom_encoding else []) + list(encodings)
    if "latin-1" not in trial:  # always keep a can't-fail fallback last
        trial.append("latin-1")

    # Try reading the file with each encoding until one decodes cleanly.
    for encoding in trial:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                lines = f.readlines()
            break
        except (UnicodeError, LookupError):
            continue
    else:
        raise EncodingError(
            f"Could not decode '{file_path}' with any of: {', '.join(trial)}."
        )

    # Keep only non-empty, non-comment lines (from the bottom up)
    valid_lines = [
        line
        for line in reversed(lines)
        if line.strip() and not line.strip().startswith("#")
    ]
    lines = list(reversed(valid_lines[:sample_lines]))
    if not lines:
        raise EmptyFileError(
            f"No usable (non-empty, non-comment) lines in '{file_path}' to "
            "analyse for delimiter detection."
        )

    best_delim: str | None = None
    best_score = 0

    # Count occurrences of each candidate delimiter per line and score consistency
    for delim in delimiters:
        counts = [line.count(delim) for line in lines]
        if not counts:
            continue

        mode = Counter(counts).most_common(1)[0][0]
        consistent = [c for c in counts if abs(c - mode) <= tolerance and c > 0]

        if len(consistent) >= min_count_threshold:
            score = len(consistent)
            if score > best_score:
                best_score = score
                best_delim = delim

    if best_delim is None:
        raise DelimiterDetectionError(
            f"Could not reliably detect a field delimiter in '{file_path}' from "
            f"the sampled lines (tried: {', '.join(repr(d) for d in delimiters)})."
        )

    return encoding, best_delim  # type: ignore[name-defined]


###############################################################################


def file_list(
    path: str,
    search_word: str | None = None,
    max_subfolder: int = 0,
    nested_list: bool = False,
) -> List[Union[str, List[str]]]:
    """List files in a folder tree with optional depth and name filtering.

    Args:
        path (str):
            Root directory to search for files.
        search_word (str | None, optional):
            Substring that must be present in the **basename** of a file for it
            to be included in the result. If ``None`` (default), all files are
            considered.
        max_subfolder (int, optional):
            Maximum depth of subfolders to traverse relative to ``path``:

            - ``0`` (default): only the root folder.
            - ``1``: root + its immediate subfolders.
            - ``2``: root + two levels down, etc.

        nested_list (bool, optional):
            If ``False`` (default), return a flat list of file paths.
            If ``True``, return a list of lists, where each inner list contains
            the files found in one directory level.

    Returns:
        list[str] | list[list[str]]:
            If ``nested_list=False``, a flat list of matching file paths.
            If ``nested_list=True``, a nested list of file paths grouped by
            directory.

    Raises:
        FileNotFoundError:
            If ``path`` does not exist. Check that the directory path is
            correct and accessible.

    Notes:
        Detailed description:
            ``file_list`` wraps :func:`os.walk` to provide a simple way to
            collect file paths from a directory tree. It:

            - Computes the depth of each directory relative to the root and
              skips directories deeper than ``max_subfolder``.
            - Builds full paths to each file using :func:`os.path.join`.
            - Filters files by ``search_word`` applied to the basename
              (filename only, without directory).
            - Returns either a flat list of paths or a nested list grouped
              by directory, depending on ``nested_list``.

            This is mainly a convenience helper used by higher-level loading
            utilities.

    Examples:
        Typical usage is to quickly collect data files to be processed
        further, e.g. before batch loading:

        .. code-block:: python

            import aerosoltools as at

            # Flat list of all CSV files in the root folder only
            csv_files = file_list(
                path="C:/data/measurements",
                search_word=".csv",
                max_subfolder=0,
            )

            # Nested list of all files up to one subfolder deep
            all_files_nested = file_list(
                path="C:/data/measurements",
                max_subfolder=1,
                nested_list=True,
            )
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    files: list[Union[str, list[str]]] = []
    root_depth = path.rstrip(os.sep).count(os.sep)

    # Walk the directory tree and respect the max_subfolder depth
    for root, _, filenames in os.walk(path):
        current_depth = root.count(os.sep)
        if current_depth - root_depth > max_subfolder:
            continue

        file_paths = [os.path.join(root, f) for f in filenames]

        if nested_list:
            if search_word:
                filtered = [f for f in file_paths if search_word in os.path.basename(f)]
                if filtered:
                    files.append(filtered)
            else:
                if file_paths:
                    files.append(file_paths)
        else:
            for f in file_paths:
                if search_word:
                    if search_word in os.path.basename(f):
                        files.append(f)
                else:
                    files.append(f)

    return files


###############################################################################


def _print_load_summary(summary_log: list[dict[str, str]]) -> None:
    """Print a compact table summarizing which files were loaded or skipped."""
    if not summary_log:
        return

    # Sort: loaded first, then skipped grouped by reason
    def sort_key(entry: dict[str, str]) -> tuple[int, str, str]:
        status = entry.get("status", "Skipped")
        reason = entry.get("reason", "")
        file = entry.get("file", "")
        return (0 if status == "Loaded" else 1, reason, file)

    sorted_entries = sorted(summary_log, key=sort_key)

    rows = [[e["file"], e["status"], e["reason"]] for e in sorted_entries]

    print("\nFile load summary:\n")
    print(
        tabulate(
            rows,
            headers=["File", "Status", "Reason"],
            tablefmt="pretty",
        )
    )


###############################################################################


def _duplicate_remover(combined_data: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate timestamps from a time-indexed DataFrame.

    This helper treats the index as a time axis, removes duplicate
    timestamps (keeping the first occurrence), and returns the result
    sorted chronologically.

    Args:
        combined_data:
            DataFrame whose index represents timestamps (or can be interpreted
            as such). The index is temporarily moved to a ``"Datetime"`` column
            while duplicates are removed.

    Returns:
        pandas.DataFrame:
            A cleaned DataFrame with a unique, sorted datetime index and no
            duplicate timestamps.

    Notes:
        - Only the first row for each duplicated timestamp is preserved.
        - The index name is set to ``"Datetime"`` in the result.
    """
    # Expose index as a column, de-duplicate, then restore as index
    combined_data = combined_data.reset_index(names=["Datetime"])
    combined_data.drop_duplicates(
        subset="Datetime", keep="first", inplace=True, ignore_index=True
    )
    combined_data.set_index("Datetime", inplace=True)
    return combined_data.sort_index()


###############################################################################


def load_data_from_folder(
    folder_path: str,
    load_function: Callable[..., "Aerosol1D"],
    search_word: str = "",
    max_subfolder: int = 0,
    meta_checklist: Sequence[str] | None = None,
    time_rebin: str | None = None,
    **kwargs: Any,
):
    """Batch-load aerosol data from multiple files in a folder, check
    metadata consistency, and combine compatible datasets into a single
    aerosol object, while skipping files that do not match.

    Args:
        folder_path (str):
            Path to the folder containing the data files to be loaded.
        load_function (Callable[..., Aerosol1D]):
            Function that loads a single file and returns an aerosol object,
            typically an instance of :class:`Aerosol1D`,
            :class:`Aerosol2D`, or an instrument subclass.
        search_word (str, optional):
            Substring that must appear in the filename for a file to be
            considered. If an empty string (default), all files in the
            directory tree (subject to ``max_subfolder``) are considered.
        max_subfolder (int, optional):
            Maximum depth of subfolders to traverse relative to ``folder_path``:

            - ``0`` (default): only the base folder.
            - ``1``: base + its immediate subfolders.
            - ``2``: base + two levels down, etc.
        meta_checklist (Sequence[str] | None, optional):
            Iterable of metadata keys that must match across all loaded files
            (e.g. ``(\"serial_number\",)``). Files with values that differ from
            the first successfully loaded file for any of these keys are
            skipped. If ``None`` (default), this is treated as
            ``(\"serial_number\",)``.
        time_rebin (str | None, optional):
            Optional resampling rule (e.g. ``"30s"``, ``"1min"``, ``"2H"``,
            ``"1D"``). If provided, each loaded object is time-rebinned via
            :meth:`Aerosol1D.timerebin` before concatenation. This can reduce
            memory load when combining many files. If ``None`` (default),
            the native time resolution of each file is preserved.
        **kwargs:
            Additional keyword arguments forwarded directly to
            ``load_function`` for each file (for example
            ``extra_data=True`` or loader-specific options).

    Returns:
        Aerosol1D | Aerosol2D:
            A combined aerosol object of the same concrete class as the first
            successfully loaded file. The returned object has:

    Raises:
        FileNotFoundError:
            If no files matching the given criteria are found in
            ``folder_path``, or if all candidate files fail to load.
            Check the folder path, search pattern, and file permissions.
        TypeError:
            If ``load_function`` returns an object that is not an instance of
            :class:`Aerosol1D`, :class:`Aerosol2D`, or an instrument subclass.
        RuntimeError:
            If an internal inconsistency occurs during concatenation (e.g.
            ``Combined_raw_data`` unexpectedly remains ``None``). This should
            not happen under normal use; if it does, please report the issue
            with a minimal reproducible example.

    Notes:
        Detailed description:
            ``load_data_from_folder`` automates the otherwise repetitive task
            of loading many similarly formatted files, checking that they
            belong together, and combining them into one continuous time
            series. Internally, it:

            - Uses :func:`file_list` to collect candidate files from
              ``folder_path`` respecting ``search_word`` and ``max_subfolder``.
            - Iterates over the candidate files with a progress bar,
              calling ``load_function(file_path, **kwargs)`` for each file.
            - Optionally calls ``timerebin(time_rebin)`` on each loaded object
              to reduce the native time resolution before combining.
            - For the first successfully loaded file, initializes containers
              for combined ``original_data``, ``extra_data``, and
              ``_raw_extra_data``, and stores its metadata as a baseline.
            - For subsequent files, checks all keys in ``meta_checklist`` for
              consistency with the baseline metadata; files that do not match
              are skipped and logged.
            - Concatenates all compatible datasets along the time axis for
              ``original_data``, ``extra_data``, and ``_raw_extra_data``.
            - Removes duplicate timestamps separately from each combined
              DataFrame using an internal helper, keeping the first occurrence
              for each timestamp.
            - Determines the concrete output class based on the type of the
              first loaded object (its concrete class) and
              instantiates the combined object accordingly.
            - Merges metadata entries such as ``"TEM_samples"`` if present in
              multiple files (relevant for the PartectorTEM).
            - Logs the status of each file (loaded or skipped) in a summary
              list and prints a compact table at the end.

            Files that raise common IO or parsing errors during loading are
            not fatal: they are simply skipped and reported in the final
            summary table, which includes filename, load status, and reason.

    Examples:
        A typical workflow is to load a full day, week, or campaign of
        measurements that are stored as multiple files (e.g. one per
        day or per instrument run):

        .. code-block:: python

            import aerosoltools as at

            # Combine all SMPS files for a given campaign folder
            smps_combined = at.load_data_from_folder(
                folder_path="C:/data/SMPS_campaign",
                load_function=load_smps_file,
                search_word=".txt",
                max_subfolder=1,
                meta_checklist=("serial_number", "instrument"),
                time_rebin="5min",   # reduce resolution for long campaigns
                extra_data=True,     # forward to load_smps_file
            )
    """
    meta_checklist = tuple(meta_checklist or ("serial_number",))

    # Collect candidate files from the folder tree
    files_any = file_list(folder_path, search_word, max_subfolder)
    if not files_any:
        raise FileNotFoundError("No files found matching the given criteria.")
    files: list[str] = [f for f in files_any if isinstance(f, str)]

    # Log of what happened to each file: {"file", "status", "reason"}
    summary_log: list[dict[str, str]] = []

    counter = 0
    Combined_raw_data: pd.DataFrame | None = None
    Combined_extra_data: pd.DataFrame | None = None
    Combined_raw_extra_data: pd.DataFrame | None = None
    meta: dict[str, Any] = {}
    Initial_data: Aerosol1D | None = None

    iterator = tqdm(files, desc="Loading files", unit="file")
    for file_path in iterator:
        fname = os.path.basename(file_path)  # only log the filename, not full path
        try:
            data = load_function(file_path, **kwargs)

            if isinstance(data, list):
                # Some loaders (e.g. Ranger files with several measurement
                # heads) return multiple objects for one file. These cannot be
                # concatenated into a single series here; load them individually.
                raise ValueError(
                    "Loader returned multiple datasets for this file; load it "
                    "individually rather than via load_data_from_folder."
                )

            if time_rebin:
                # Rebin the data in time to reduce resolution and memory usage
                data.timerebin(time_rebin)
                # Keep a copy of the rebinned data as the "raw" reference
                data._raw_data = data._data
                data._raw_extra_data = data._extra_data

            if counter == 0:
                # Initialize combined containers from the first successful file
                Initial_data = data
                meta = dict(data.metadata)
                Combined_raw_data = data.original_data
                Combined_extra_data = data.extra_data
                Combined_raw_extra_data = data._raw_extra_data
                counter = 1
                summary_log.append({"file": fname, "status": "Loaded", "reason": "-"})
            else:
                # Verify that required metadata entries match
                mismatch_found = False
                for item in meta_checklist:
                    if data.metadata.get(item) != meta.get(item):
                        summary_log.append(
                            {
                                "file": fname,
                                "status": "Skipped",
                                "reason": f"Metadata mismatch: {item}",
                            }
                        )
                        mismatch_found = True
                        break

                if mismatch_found:
                    continue

                # Append compatible data
                Combined_raw_data = pd.concat(  # type: ignore[arg-type]
                    [Combined_raw_data, data.original_data]
                )
                Combined_extra_data = pd.concat(  # type: ignore[arg-type]
                    [Combined_extra_data, data.extra_data]
                )
                Combined_raw_extra_data = pd.concat(  # type: ignore[arg-type]
                    [Combined_raw_extra_data, data._raw_extra_data]
                )

                # Optionally merge TEM sample information if present
                if "TEM_samples" in data.metadata:
                    if "TEM_samples" in meta:
                        meta["TEM_samples"] = pd.concat(
                            [meta["TEM_samples"], data.metadata["TEM_samples"]]
                        )
                    else:
                        meta["TEM_samples"] = data.metadata["TEM_samples"]

                summary_log.append({"file": fname, "status": "Loaded", "reason": "-"})

        except (
            LoaderError,
            FileNotFoundError,
            ValueError,
            KeyError,
            UnicodeDecodeError,
            TypeError,
            IndexError,
        ) as e:
            summary_log.append(
                {
                    "file": fname,
                    "status": "Skipped",
                    "reason": f"{type(e).__name__}: {e}",
                }
            )

    # ---- Post-loop guards / narrowing -------------------------------------

    if Initial_data is None:
        _print_load_summary(summary_log)
        raise FileNotFoundError(
            "No valid files found in folder (or all files failed to load)."
        )

    if Combined_raw_data is None:
        _print_load_summary(summary_log)
        raise RuntimeError("Internal error: Combined_raw_data is None after loading.")

    # Remove duplicate timestamps from each combined table
    Combined_raw_data = _duplicate_remover(Combined_raw_data)

    if Combined_extra_data is not None:
        Combined_extra_data = _duplicate_remover(Combined_extra_data)
    else:
        Combined_extra_data = pd.DataFrame(index=Combined_raw_data.index)

    if Combined_raw_extra_data is not None:
        Combined_raw_extra_data = _duplicate_remover(Combined_raw_extra_data)
    else:
        Combined_raw_extra_data = pd.DataFrame(index=Combined_raw_data.index)

    # Instantiate the appropriate concrete type based on the first file. Size-
    # resolved data is rebuilt as Aerosol2D (dual-distribution Aerosol3d is
    # flattened, as before). Any 1D family object is rebuilt as its *own*
    # concrete type via ``type(Initial_data)`` so the non-particle classes
    # (Gas1D / Aethalometer / DiSCmini / DustTrak / ...) and plain Aerosol1D
    # survive combining instead of collapsing to a base class.
    if isinstance(Initial_data, Aerosol2D):
        Combined_data: Aerosol1D = Aerosol2D(Combined_raw_data)
    elif isinstance(Initial_data, Aerosol1D):
        Combined_data = type(Initial_data)(Combined_raw_data)
    else:
        _print_load_summary(summary_log)
        raise TypeError("Unsupported data type returned by load_function.")

    Combined_data._extra_data = Combined_extra_data
    Combined_data._raw_extra_data = Combined_raw_extra_data
    Combined_data._meta = meta

    _print_load_summary(summary_log)

    return Combined_data
