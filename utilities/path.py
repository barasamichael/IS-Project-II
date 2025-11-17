"""
Path utilities for the International Student Assistant system.
Provides bulletproof path handling and validation.
"""
import os
import logging

from pathlib import Path
from functools import wraps

from typing import List
from typing import Union
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("path_utils")


def ensure_path(path_input: Union[str, Path, None]) -> Optional[Path]:
    """
    Ensure input is converted to Path object with proper error handling.

    Args:
        path_input: String path, Path object, or None

    Returns:
        Path object or None if input was None

    Raises:
        ValueError: If path_input is not a valid path type
    """
    if path_input is None:
        return None

    if isinstance(path_input, Path):
        return path_input

    if isinstance(path_input, str):
        if not path_input.strip():
            return None
        try:
            return Path(path_input)
        except Exception as e:
            logger.error(f"Invalid path string '{path_input}': {str(e)}")
            raise ValueError(f"Cannot convert '{path_input}' to Path: {str(e)}")

    raise ValueError(
        f"Invalid path type: {type(path_input)}. Expected str, Path, or None."
    )


def safe_path_operation(func):
    """
    Decorator to safely handle path operations by converting string arguments to Path objects.
    Automatically detects path-like arguments and converts them.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Convert string paths to Path objects in args
        new_args = []
        for arg in args:
            if isinstance(arg, str) and _looks_like_path(arg):
                new_args.append(ensure_path(arg))
            else:
                new_args.append(arg)

        # Convert string paths to Path objects in kwargs
        new_kwargs = {}
        for key, value in kwargs.items():
            if _is_path_parameter(key) and isinstance(value, str):
                new_kwargs[key] = ensure_path(value)
            else:
                new_kwargs[key] = value

        return func(*new_args, **new_kwargs)

    return wrapper


def _looks_like_path(arg: str) -> bool:
    """Heuristically determine if a string looks like a file path."""
    if not arg or not isinstance(arg, str):
        return False

    # Check for path separators
    if "/" in arg or "\\" in arg:
        return True

    # Check for common file extensions
    file_extensions = {
        ".txt",
        ".json",
        ".jsonl",
        ".npz",
        ".md",
        ".pdf",
        ".docx",
        ".csv",
        ".xlsx",
        ".html",
        ".htm",
        ".xml",
        ".yaml",
        ".yml",
        ".py",
        ".js",
        ".css",
        ".log",
        ".tmp",
        ".bak",
    }

    if any(arg.lower().endswith(ext) for ext in file_extensions):
        return True

    # Check for common directory names
    dir_names = {"data", "uploads", "downloads", "documents", "temp", "tmp"}
    if arg.lower() in dir_names:
        return True

    return False


def _is_path_parameter(param_name: str) -> bool:
    """Check if a parameter name suggests it contains a path."""
    path_indicators = [
        "_path",
        "_dir",
        "_file",
        "_folder",
        "_directory",
        "path",
        "file_path",
        "dir_path",
        "output_path",
        "input_path",
        "chunks_file",
        "embeddings_file",
        "raw_dir",
        "processed_dir",
    ]

    return any(indicator in param_name.lower() for indicator in path_indicators)


def validate_directory(
    directory: Union[str, Path], create_if_missing: bool = False
) -> Path:
    """
    Validate that a directory exists and is accessible.

    Args:
        directory: Directory path to validate
        create_if_missing: Whether to create the directory if it doesn't exist

    Returns:
        Validated Path object

    Raises:
        ValueError: If directory is invalid or inaccessible
    """
    dir_path = ensure_path(directory)
    if not dir_path:
        raise ValueError("Directory path cannot be None or empty")

    try:
        if not dir_path.exists():
            if create_if_missing:
                dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {dir_path}")
            else:
                raise ValueError(f"Directory does not exist: {dir_path}")

        if not dir_path.is_dir():
            raise ValueError(f"Path exists but is not a directory: {dir_path}")

        # Test write access
        test_file = dir_path / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            raise ValueError(
                f"Directory is not writable: {dir_path} - {str(e)}"
            )

        return dir_path

    except Exception as e:
        logger.error(f"Directory validation failed for {dir_path}: {str(e)}")
        raise ValueError(f"Invalid directory {dir_path}: {str(e)}")


def validate_file(file_path: Union[str, Path], must_exist: bool = True) -> Path:
    """
    Validate that a file exists and is accessible.

    Args:
        file_path: File path to validate
        must_exist: Whether the file must exist

    Returns:
        Validated Path object

    Raises:
        ValueError: If file is invalid or inaccessible
    """
    file_path = ensure_path(file_path)
    if not file_path:
        raise ValueError("File path cannot be None or empty")

    try:
        if must_exist:
            if not file_path.exists():
                raise ValueError(f"File does not exist: {file_path}")

            if not file_path.is_file():
                raise ValueError(f"Path exists but is not a file: {file_path}")

            # Test read access
            try:
                with open(file_path, "r") as f:
                    f.read(1)  # Try to read one character
            except Exception as e:
                raise ValueError(
                    f"File is not readable: {file_path} - {str(e)}"
                )
        else:
            # For files that don't need to exist, validate the parent directory
            parent_dir = file_path.parent
            if not parent_dir.exists():
                raise ValueError(
                    f"Parent directory does not exist: {parent_dir}"
                )

        return file_path

    except Exception as e:
        logger.error(f"File validation failed for {file_path}: {str(e)}")
        raise ValueError(f"Invalid file {file_path}: {str(e)}")


def safe_file_operation(operation: str):
    """
    Decorator for safe file operations with automatic cleanup.

    Args:
        operation: Type of operation ('read', 'write', 'append')
    """

    def decorator(func):
        @wraps(func)
        def wrapper(file_path: Union[str, Path], *args, **kwargs):
            file_path = ensure_path(file_path)

            # Validate file based on operation
            if operation in ["read"]:
                validate_file(file_path, must_exist=True)
            elif operation in ["write", "append"]:
                validate_directory(file_path.parent, create_if_missing=True)

            try:
                return func(file_path, *args, **kwargs)
            except Exception as e:
                logger.error(
                    f"File operation '{operation}' failed for {file_path}: {str(e)}"
                )
                raise

        return wrapper

    return decorator


def get_file_info(file_path: Union[str, Path]) -> dict:
    """
    Get comprehensive information about a file.

    Args:
        file_path: Path to the file

    Returns:
        Dictionary with file information
    """
    file_path = ensure_path(file_path)
    if not file_path or not file_path.exists():
        return {"exists": False, "error": "File not found"}

    try:
        stat = file_path.stat()

        return {
            "exists": True,
            "path": str(file_path),
            "name": file_path.name,
            "stem": file_path.stem,
            "suffix": file_path.suffix,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "accessed": stat.st_atime,
            "is_file": file_path.is_file(),
            "is_directory": file_path.is_dir(),
            "is_readable": os.access(file_path, os.R_OK),
            "is_writable": os.access(file_path, os.W_OK),
            "absolute_path": str(file_path.absolute()),
        }

    except Exception as e:
        return {"exists": True, "error": str(e)}


def clean_filename(filename: str, replacement: str = "_") -> str:
    """
    Clean a filename by removing or replacing invalid characters.

    Args:
        filename: Original filename
        replacement: Character to replace invalid characters with

    Returns:
        Cleaned filename
    """
    if not filename:
        return "unnamed_file"

    # Remove or replace invalid characters for cross-platform compatibility
    invalid_chars = ["<", ">", ":", '"', "/", "\\", "|", "?", "*"]

    cleaned = filename
    for char in invalid_chars:
        cleaned = cleaned.replace(char, replacement)

    # Remove leading/trailing whitespace and dots
    cleaned = cleaned.strip(". ")

    # Limit length
    if len(cleaned) > 255:
        name, ext = os.path.splitext(cleaned)
        max_name_length = 255 - len(ext)
        cleaned = name[:max_name_length] + ext

    return cleaned or "unnamed_file"


def find_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = True,
    max_results: Optional[int] = None,
) -> List[Path]:
    """
    Find files matching a pattern in a directory.

    Args:
        directory: Directory to search in
        pattern: Glob pattern to match
        recursive: Whether to search recursively
        max_results: Maximum number of results to return

    Returns:
        List of matching file paths
    """
    directory = ensure_path(directory)
    if not directory or not directory.exists():
        logger.warning(f"Directory not found for search: {directory}")
        return []

    try:
        if recursive:
            files = list(directory.rglob(pattern))
        else:
            files = list(directory.glob(pattern))

        # Filter to only include files (not directories)
        files = [f for f in files if f.is_file()]

        # Sort by modification time (newest first)
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        # Limit results if specified
        if max_results:
            files = files[:max_results]

        return files

    except Exception as e:
        logger.error(f"Error searching for files: {str(e)}")
        return []


def atomic_write(
    file_path: Union[str, Path], content: str, encoding: str = "utf-8"
) -> bool:
    """
    Atomically write content to a file using a temporary file.

    Args:
        file_path: Destination file path
        content: Content to write
        encoding: File encoding

    Returns:
        True if successful, False otherwise
    """
    file_path = ensure_path(file_path)
    if not file_path:
        return False

    # Create temporary file in the same directory
    temp_path = file_path.parent / f".{file_path.name}.tmp"

    try:
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to temporary file
        with open(temp_path, "w", encoding=encoding) as f:
            f.write(content)

        # Atomic rename (works on most filesystems)
        temp_path.replace(file_path)

        logger.debug(
            f"Atomically wrote {len(content)} characters to {file_path}"
        )
        return True

    except Exception as e:
        logger.error(f"Atomic write failed for {file_path}: {str(e)}")
        # Clean up temporary file if it exists
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass
        return False


def backup_file(
    file_path: Union[str, Path], backup_suffix: str = ".bak"
) -> Optional[Path]:
    """
    Create a backup copy of a file.

    Args:
        file_path: File to backup
        backup_suffix: Suffix for backup file

    Returns:
        Path to backup file if successful, None otherwise
    """
    file_path = ensure_path(file_path)
    if not file_path or not file_path.exists():
        logger.warning(f"Cannot backup non-existent file: {file_path}")
        return None

    backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)

    try:
        # Use shutil.copy2 to preserve metadata
        import shutil

        shutil.copy2(file_path, backup_path)

        logger.info(f"Created backup: {backup_path}")
        return backup_path

    except Exception as e:
        logger.error(f"Backup failed for {file_path}: {str(e)}")
        return None


def get_disk_usage(directory: Union[str, Path]) -> dict:
    """
    Get disk usage statistics for a directory.

    Args:
        directory: Directory to check

    Returns:
        Dictionary with usage statistics
    """
    directory = ensure_path(directory)
    if not directory or not directory.exists():
        return {"error": "Directory not found"}

    try:
        import shutil

        total, used, free = shutil.disk_usage(directory)

        return {
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": free,
            "total_gb": round(total / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "usage_percentage": round((used / total) * 100, 1)
            if total > 0
            else 0,
        }

    except Exception as e:
        return {"error": str(e)}


# Constants for common file types in the international student system
SUPPORTED_DOCUMENT_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".doc": "application/msword",
    ".html": "text/html",
    ".htm": "text/html",
}

SUPPORTED_DATA_TYPES = {
    ".json": "application/json",
    ".jsonl": "application/jsonlines",
    ".csv": "text/csv",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".npz": "application/octet-stream",
}

ALL_SUPPORTED_TYPES = {**SUPPORTED_DOCUMENT_TYPES, **SUPPORTED_DATA_TYPES}


def is_supported_file_type(file_path: Union[str, Path]) -> bool:
    """Check if a file type is supported by the system."""
    file_path = ensure_path(file_path)
    if not file_path:
        return False

    return file_path.suffix.lower() in ALL_SUPPORTED_TYPES


def get_file_type_info(file_path: Union[str, Path]) -> dict:
    """Get file type information."""
    file_path = ensure_path(file_path)
    if not file_path:
        return {"supported": False, "error": "Invalid path"}

    suffix = file_path.suffix.lower()
    supported = suffix in ALL_SUPPORTED_TYPES

    return {
        "suffix": suffix,
        "supported": supported,
        "mime_type": ALL_SUPPORTED_TYPES.get(suffix, "unknown"),
        "category": "document"
        if suffix in SUPPORTED_DOCUMENT_TYPES
        else "data"
        if suffix in SUPPORTED_DATA_TYPES
        else "unknown",
    }
