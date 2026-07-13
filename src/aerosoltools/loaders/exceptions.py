"""Exception hierarchy for the aerosoltools loader layer.

Every error raised while loading an instrument file derives from
:class:`LoaderError`, so a caller can catch the whole family with one
``except LoaderError``. The subclasses name the failing stage precisely
(access, encoding, delimiter, empty, format, metadata, timestamp, detection) so
messages are actionable — a batch loader can report *why* a file was skipped,
and a script can branch on the specific failure.
"""

from __future__ import annotations


class LoaderError(Exception):
    """Base class for every error raised while loading an instrument file."""


class FileAccessError(LoaderError):
    """The file could not be found, opened, or read."""


class EmptyFileError(LoaderError):
    """The file has no usable (non-empty, non-comment) data lines."""


class EncodingError(LoaderError):
    """The file could not be decoded with any candidate text encoding."""


class DelimiterDetectionError(LoaderError):
    """No consistent field delimiter could be inferred from the file."""


class FileFormatError(LoaderError):
    """The file does not match the format the loader expects.

    Raised when required header markers, columns, or structure are missing —
    i.e. the file is readable text but not the instrument export it should be.
    """


class MetadataParseError(LoaderError):
    """A required metadata field could not be parsed from the file header."""


class TimestampError(LoaderError):
    """Timestamps could not be parsed, or are missing/inconsistent."""


class InstrumentDetectionError(LoaderError):
    """The instrument type could not be identified from content or filename."""


__all__ = [
    "LoaderError",
    "FileAccessError",
    "EmptyFileError",
    "EncodingError",
    "DelimiterDetectionError",
    "FileFormatError",
    "MetadataParseError",
    "TimestampError",
    "InstrumentDetectionError",
]
