import logging
import os
import re
from typing import List, Union

logger = logging.getLogger(__name__)

# Direct copy from schedtools.compression. Reproduced here to avoid dependency.

def build_zip(
    zip_file: str, path: str, exclude_patterns: Union[str, List[str], None] = None
) -> None:
    """Add files to a zip archive.

    Args:
        zip_file: Path to the zip file
        path: Path to the directory to zip
    """
    import zipfile

    if exclude_patterns is None:
        exclude_patterns = []
    elif isinstance(exclude_patterns, str):
        exclude_patterns = [exclude_patterns]
    exclude_compiled = [re.compile(pattern) for pattern in exclude_patterns]

    logger.info(f"Zipping {path} into {zip_file}.")

    with zipfile.ZipFile(zip_file, "a") as zf:
        existing_files = set(zf.namelist())
        for root, _, files in os.walk(path):
            for file in files:
                if any(pattern.search(file) for pattern in exclude_compiled):
                    logger.debug(f"Skipping '{file}' in '{path}'.")
                    continue
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, path)
                if arcname in existing_files:
                    logger.debug(f"Skipping duplicate '{arcname}' in '{path}'.")
                    continue
                zf.write(file_path, arcname)


def zip_experiment(path: str):
    zip_file = path + ".zip"
    build_zip(zip_file, path, exclude_patterns=[r"\.pbs\.e\d+", r"\.pbs\.o"])
