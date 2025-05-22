# -*- coding: utf-8 -*-

import os
from collections import Counter
from typing import List, Union

import pandas as pd

from ..aerosol1d import Aerosol1D
from ..aerosol2d import Aerosol2D
from ..aerosolalt import AerosolAlt

###############################################################################


def detect_delimiter(
    file_path: str,
    encodings: list = ["latin-1", "utf-8", "utf-16", "iso-8859-1", "windows-1252"],
    delimiters: list = [",", ";", "\t", "|"],
    sample_lines: int = 10,
    min_count_threshold: int = 3,
    tolerance: int = 1,
):
    """
    Automatically detect the encoding and delimiter of a delimited text file.

    This function attempts to read the file using multiple encodings, and then
    tests a range of delimiters to determine the one with the most consistent
    occurrence across sampled lines. It ignores empty lines and comment lines
    starting with '#'.

    Parameters
    ----------
    file_path : str
        Path to the input text or CSV-like file.
    encodings : list of str, optional
        List of character encodings to try. Default includes common options.
    delimiters : list of str, optional
        List of possible field delimiters. Default is [',', ';', '\\t', '|'].
    sample_lines : int, optional
        Number of non-empty lines to analyze from the top of the file. Default is 10.
    min_count_threshold : int, optional
        Minimum number of lines that must show consistent delimiter usage.
        Default is 3.
    tolerance : int, optional
        Allowed deviation from modal delimiter count across lines. Default is 1.

    Returns
    -------
    encoding : str
        Detected encoding that successfully opened the file.
    delimiter : str
        Most consistent delimiter found based on column counts.

    Raises
    ------
    UnicodeDecodeError
        If no encoding in the list allows the file to be read.
    ValueError
        If no reliable delimiter could be detected from sampled lines.

    Examples
    --------
    >>> detect_delimiter("data.csv")
    ('utf-8', ',')

    Notes
    -----
    - This function is helpful for preprocessing arbitrary files without header info.
    - You can tune the sensitivity by adjusting `sample_lines`, `min_count_threshold`, and `tolerance`.
    """
    # Try reading file with multiple encodings
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as f:
                lines = f.readlines()
            break
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("Could not decode file with given encodings.")
    # Filter non-empty, non-comment lines from the bottom
    valid_lines = [
        line
        for line in reversed(lines)
        if line.strip() and not line.strip().startswith("#")
    ]
    lines = list(reversed(valid_lines[:sample_lines]))  # Keep original order
    if not lines:
        raise ValueError("No valid lines found to analyze.")

    best_delim = None
    best_score = 0

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

    if best_delim:
        return encoding, best_delim
    else:
        raise ValueError("Could not reliably detect a delimiter.")


###############################################################################


def file_list(
    path: str,
    search_word: Union[str, None] = None,
    max_subfolder: int = 0,
    nested_list: bool = False,
) -> List[Union[str, List[str]]]:
    """
    Generate a list of file paths from a directory, with optional search filtering and folder nesting.

    Parameters
    ----------
    path : str
        Root directory to search for files.
    search_word : str or None, optional
        If provided, only include files containing this substring in their filenames.
        Default is None (includes all files).
    max_subfolder : int, optional
        Maximum depth of subfolders to include (0 = current folder only).
        Default is 0.
    nested_list : bool, optional
        If True, returns a list of lists (one list per subdirectory).
        If False, returns a flat list of all matching file paths. Default is False.

    Returns
    -------
    List[str] or List[List[str]]
        Flat list of file paths, or nested list of file paths if `nested_list=True`.

    Examples
    --------
    >>> file_list("/data/logs", search_word="2024", max_subfolder=1)
    ['/data/logs/log_2024.txt', '/data/logs/archive/log_2024_summary.csv']

    >>> file_list("/data", nested_list=True)
    [['/data/a.txt', '/data/b.txt'], ['/data/subdir/c.txt']]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Path does not exist: {path}")

    files = []
    root_depth = path.rstrip(os.sep).count(os.sep)

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


def duplicate_remover(combined_data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate entries based on the datetime index in a time series DataFrame.

    This function resets the index to expose 'Datetime' as a column, removes duplicate
    timestamps (keeping the first occurrence), then restores the datetime index and sorts it.

    Parameters
    ----------
    combined_data : pd.DataFrame
        A DataFrame with a DatetimeIndex or an index to be treated as timestamps.

    Returns
    -------
    pd.DataFrame
        A cleaned and chronologically sorted DataFrame with duplicates removed.

    Notes
    -----
    - Only the first occurrence of a duplicated datetime is retained.
    - Index is expected to represent datetime values.

    Examples
    --------
    >>> cleaned = duplicate_remover(raw_data)
    >>> print(cleaned.index.is_unique)  # True
    """
    combined_data = combined_data.reset_index(names=["Datetime"])
    combined_data.drop_duplicates(
        subset="Datetime", keep="first", inplace=True, ignore_index=True
    )
    combined_data.set_index("Datetime", inplace=True)
    return combined_data.sort_index()


###############################################################################


def Load_data_from_folder(
    folder_path,
    load_function,
    search_word="",
    max_subfolder=0,
    meta_checklist: list = ["serial_number"],
    **kwargs,
):
    """
    Load and concatenate aerosol data from a folder using a specified loader function.

    This function iterates over all files in the specified folder (and optionally its
    subfolders) that match a given search word. Each file is processed using the
    provided load_function. Files with incompatible metadata (based on keys in
    meta_checklist) are skipped. The function returns a combined aerosol data
    object with merged time-series and metadata.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing the data files.

    load_function : function
        A function that loads a single data file and returns an instance of a class
        like Aerosol1D, Aerosol2D, or AerosolAlt. The function must return an object
        with original_data, extra_data, and metadata.

    search_word : str, optional
        A string that must be present in the filename for the file to be loaded.
        Defaults to "" (match all files).

    max_subfolder : int, optional
        Depth of subfolder levels to include in the search.
        0 means only the base folder is used; 1 includes immediate subfolders, etc.

    meta_checklist : list of str, optional
        List of metadata keys that must be identical across all loaded files.
        If any key differs, the file is skipped. Defaults to ["serial_number"].

    kwargs
        Additional keyword arguments passed to the load_function.

    Returns
    -------
    Combined_data : Aerosol1D or Aerosol2D or AerosolAlt
        A combined aerosol data object. The returned object inherits from the same
        class as the first successfully loaded file. It includes:
        
        - Combined original_data
        - Combined extra_data
        - Merged metadata

    Notes
    -----
    Files that raise exceptions or fail metadata consistency checks are skipped.
    A message will be printed for each skipped file, along with the reason.
    The function will raise an exception if no valid files are found or if the returned
    object is not an instance of Aerosol1D, Aerosol2D, or AerosolAlt.
    """

    counter = 0
    skipped_files = []
    Combined_raw_data = None
    Combined_extra_data = None
    meta = {}

    for file_path in file_list(folder_path, search_word, max_subfolder):
        print(f"Loading: {file_path}")
        try:
            data = load_function(file_path, **kwargs)

            if counter == 0:
                Initial_data = data
                meta = data.metadata
                Combined_raw_data = data.original_data
                Combined_extra_data = data.extra_data
                counter = 1
            else:
                # Check metadata consistency
                mismatch_found = False
                for item in meta_checklist:
                    if data.metadata.get(item) != meta.get(item):
                        print(f"unequal {item}")
                        skipped_files.append(file_path)
                        mismatch_found = True
                        break

                if not mismatch_found:
                    Combined_raw_data = pd.concat(
                        [Combined_raw_data, data.original_data]
                    )
                    Combined_extra_data = pd.concat(
                        [Combined_extra_data, data.extra_data]
                    )

                    if "TEM_samples" in data.metadata:
                        if "TEM_samples" in meta:
                            meta["TEM_samples"] = pd.concat(
                                [meta["TEM_samples"], data.metadata["TEM_samples"]]
                            )
                        else:
                            meta["TEM_samples"] = data.metadata["TEM_samples"]

        except (
            FileNotFoundError,
            ValueError,
            KeyError,
            UnicodeDecodeError,
            TypeError,
        ) as e:
            print(f"Skipping {file_path} due to error: {type(e).__name__}: {e}")
            skipped_files.append(file_path)

    if Combined_raw_data is not None:
        Combined_raw_data = duplicate_remover(Combined_raw_data)
    if Combined_extra_data is not None:
        Combined_extra_data = duplicate_remover(Combined_extra_data)

    # Instantiate final data object based on original class
    if isinstance(Initial_data, Aerosol2D):
        Combined_data = Aerosol2D(Combined_raw_data)
    elif isinstance(Initial_data, AerosolAlt):
        Combined_data = AerosolAlt(Combined_raw_data)
    elif isinstance(Initial_data, Aerosol1D):
        Combined_data = Aerosol1D(Combined_raw_data)
    else:
        raise Exception("Unsupported data type returned by load_function")

    Combined_data._extra_data = Combined_extra_data
    Combined_data._meta = meta

    if skipped_files:
        print("Files skipped due to errors or empty datasets:")
        for i in skipped_files:
            print(i)

    return Combined_data
