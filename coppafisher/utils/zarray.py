import os
import shutil
import subprocess
import sys
import tempfile

import zarr
from numcodecs import blosc

from . import cli


def set_zarr_global_configs() -> None:
    """
    Set any zarr global configurations before being used.
    """
    blosc.use_threads = True
    blosc.set_nthreads(16)


def image_exists(file_path: str) -> bool:
    """
    Checks if an zarr.Array exists at the given path location.

    Args:
        file_path (str): tile path.

    Returns:
        bool: tile existence.
    """
    return os.path.isfile(file_path)


def convert_group_to_zip_store(group_path: str, temp_directory: str) -> None:
    """
    Store a zarr.Group into a ZipStore.

    It is zipped and then stored at the same location.

    Args:
        group_path (str): the file path to the zarr Group.
        temp_directory (str, optional): the directory to store the zipped group temporarily. If set to "", a temporary
            directory is made using [`tempfile`](https://docs.python.org/3/library/tempfile.html).
    """
    if not os.path.exists(group_path):
        raise FileNotFoundError(f"Nothing at {group_path=}")
    if not os.path.isdir(group_path):
        raise ValueError(f"Expected {group_path=} to be a directory")
    if temp_directory:
        if not os.path.isdir(temp_directory):
            raise SystemError(f"Could not find temporary directory at {temp_directory}")
        temp_dir = None
    else:
        temp_dir = tempfile.TemporaryDirectory("coppafisher")
        temp_directory = temp_dir.name

    # Ensure that files can move between the temporary directory and the destination before continuing.
    temp_test_path = os.path.join(temp_directory, "test_move.zip")
    temp_test_destination = os.path.join(os.path.dirname(group_path), "test_move.zip")
    with open(temp_test_path, "wb"):
        pass
    assert os.path.exists(temp_test_path)
    shutil.move(temp_test_path, temp_test_destination)
    assert os.path.exists(temp_test_destination)
    os.remove(temp_test_destination)

    store = zarr.DirectoryStore(group_path)
    group = zarr.open_group(store, "r", zarr_version=2)

    temp_path = os.path.join(temp_directory, os.path.basename(group_path))
    temp_store = zarr.ZipStore(temp_path)
    temp_group = zarr.open_group(temp_store, "w-")
    zarr.copy_all(group, temp_group)
    temp_store.close()

    shutil.rmtree(group_path)
    shutil.move(temp_path, group_path)

    if temp_dir is not None:
        temp_dir.cleanup()


def convert_array_to_zip_store(array_path: str, temp_directory: str) -> None:
    """
    Store a zarr.Array into a ZipStore.

    The array is zipped and stored at the same location.

    Args:
        group_path (str): the file path to the zarr Group.
        temp_directory (str, optional): the directory to store the zipped array temporarily. If set to "", a temporary
            directory is made using [`tempfile`](https://docs.python.org/3/library/tempfile.html).
    """
    if not os.path.exists(array_path):
        raise FileNotFoundError(f"Nothing at {array_path=}")
    if not os.path.isdir(array_path):
        raise ValueError(f"Expected {array_path=} to be a directory")

    if not cli.has_cli_tool("7z"):
        msg = "Command line tool 7z was not found."
        if sys.platform == "win32":
            msg += " Install it from their website (https://www.7-zip.org/) and add the 7z.exe to your PATH."
            msg += " Or install it via Chocolatey, e.g. choco install 7zip -y."
        else:
            msg += " Install it, e.g. sudo apt-get update && sudo apt-get install -y p7zip-full"
        raise SystemError(msg)

    if temp_directory:
        if not os.path.isdir(temp_directory):
            raise SystemError(f"Could not find temporary directory at {temp_directory}")
        temp_dir = None
    else:
        temp_dir = tempfile.TemporaryDirectory("coppafisher")
        temp_directory = temp_dir.name

    # Ensure that files can move between the temporary directory and the destination before continuing.
    temp_test_path = os.path.join(temp_directory, "test_move.zip")
    temp_test_destination = os.path.join(os.path.dirname(array_path), "test_move.zip")
    with open(temp_test_path, "wb"):
        pass
    assert os.path.exists(temp_test_path)
    shutil.move(temp_test_path, temp_test_destination)
    assert os.path.exists(temp_test_destination)
    os.remove(temp_test_destination)

    temp_path = os.path.join(temp_directory, "temp_zarr_copy.zip")
    subprocess.run(["7z", "a", "-tzip", temp_path, os.path.join(array_path, ".")], capture_output=True, check=True)
    if not os.path.isfile(temp_path):
        raise FileNotFoundError(f"Failed to zip to file position {temp_path}, file was not found")
    shutil.rmtree(array_path)
    shutil.move(temp_path, array_path)

    if temp_dir is not None:
        temp_dir.cleanup()
