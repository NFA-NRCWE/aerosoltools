# -*- coding: utf-8 -*-

import pandas as pd
import os
from collections import Counter
from aerosol1d import Aerosol1D
from aerosol2d import Aerosol2D
from aerosolalt import AerosolAlt
from typing import Union, List

###############################################################################

def detect_delimiter(
    file_path: str,
    encodings: list = ['utf-8', 'iso-8859-1', 'windows-1252'],
    delimiters: list = [',', ';', '\t', '|'],
    sample_lines: int = 10,
    min_count_threshold: int = 3,
    tolerance: int = 1
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
            with open(file_path, 'r', encoding=encoding) as f:
                lines = [
                    line.strip()
                    for _, line in zip(range(sample_lines), f)
                    if line.strip() and not line.startswith('#')
                ]
            break  # success
        except UnicodeDecodeError:
            continue
    else:
        raise UnicodeDecodeError("Could not decode file with any provided encodings.")

    # Track best scoring delimiter
    best_delim = None
    best_score = 0

    # Evaluate each delimiter
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
        raise ValueError("Could not reliably detect a delimiter from the sampled lines.")

###############################################################################

def file_list(
    path: str,
    search_word: Union[str, None] = None,
    max_subfolder: int = 0,
    nested_list: bool = False
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
    combined_data = combined_data.reset_index(names=['Datetime'])
    combined_data.drop_duplicates(
        subset='Datetime',
        keep='first',
        inplace=True,
        ignore_index=True
    )
    combined_data.set_index('Datetime', inplace=True)
    return combined_data.sort_index()


###############################################################################    

def Load_data_from_folder(
    folder_path: str,
    load_function,
    search_word: str = "",
    max_subfolder: int = 0,
    meta_checklist: list = ['serial_number'],
    **kwargs
    ):
    """
    Load and concatenate multiple aerosol datasets from a folder using a specified load function.

    This function searches a directory (and optionally subdirectories) for files matching
    a given pattern, applies a custom loader to each, and concatenates the results into
    a single aerosol object. It also verifies consistency in key metadata fields.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing input files.
    load_function : function
        The loader function used to parse each file (e.g., `Load_CPC`, `Load_OPS_file`).
    search_word : str, optional
        If specified, only files containing this substring in their name will be loaded.
        Default is "" (no filter).
    max_subfolder : int, optional
        Depth of subfolder traversal. 0 = top-level only. Default is 0.
    meta_checklist : list of str, optional
        List of metadata keys to check for consistency across files. Default is ['serial_number'].
    **kwargs : dict
        Additional keyword arguments passed to the `load_function`.

    Returns
    -------
    Combined_data : aerosol1d or aerosol2d or AerosolAlt
        Concatenated and deduplicated data object with combined metadata and extra data.

    Raises
    ------
    ValueError
        If inconsistent metadata is found for required fields in `meta_checklist`.

    Notes
    -----
    - Uses `file_list()` and `duplicate_remover()` utilities.
    - Metadata like TEM samples is preserved and merged when applicable.
    """
    counter = 0
    Combined_raw_data = None
    Combined_extra_data = None

    for file_path in file_list(folder_path, search_word, max_subfolder):
        print(f"Loading: {file_path}")
        next_data = load_function(file_path, **kwargs)

        if counter == 0:
            initial_data = next_data
            meta = next_data.metadata
            Combined_raw_data = next_data.original_data
            Combined_extra_data = next_data.extra_data
            counter += 1
        else:
            for key in meta_checklist:
                if next_data.metadata.get(key) != meta.get(key):
                    print(f"Warning: metadata mismatch on '{key}'")
                    continue

            Combined_raw_data = pd.concat([Combined_raw_data, next_data.original_data])
            Combined_extra_data = pd.concat([Combined_extra_data, next_data.extra_data])

            if 'TEM_samples' in next_data.metadata:
                if 'TEM_samples' in meta:
                    meta['TEM_samples'] = pd.concat([meta['TEM_samples'], next_data.metadata['TEM_samples']])
                else:
                    meta['TEM_samples'] = next_data.metadata['TEM_samples']

    # Deduplicate timestamps
    Combined_raw_data = duplicate_remover(Combined_raw_data)
    Combined_extra_data = duplicate_remover(Combined_extra_data)

    # Rebuild using the appropriate class
    if type(initial_data) == Aerosol1D:
        Combined_data = Aerosol1D(Combined_raw_data)
    elif type(initial_data) == Aerosol2D:
        Combined_data = Aerosol2D(Combined_raw_data)
    elif type(initial_data) == AerosolAlt:
        Combined_data = AerosolAlt(Combined_raw_data)
    else:
        raise TypeError("Unsupported data class returned from load function.")

    Combined_data._extra_data = Combined_extra_data
    Combined_data._meta = meta

    return Combined_data


###############################################################################
