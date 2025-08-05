import os
import shutil
import subprocess
import sys
from pathlib import PurePath
from typing import Tuple

import numpy as np
import psutil
import torch

from . import web


class SystemConstants:
    VERSION_CONTAINS = "version"
    # The character(s) that encapsulate the software version tag in _version.py, in this case it is quotation marks.
    VERSION_ENCAPSULATE = '"'


def get_version_url() -> str:
    # This is temporary until 1.3.0 is pushed to main; then the version URL will then be a constant.
    version = get_software_version().split(".")
    if (
        version[0] == "1"
        and version[1] == "3"
        and version[2].startswith("0")
        and web.try_read_url_at("https://github.com/paulshuker/coppafisher/raw/HEAD/coppafisher/_version.py")
        is not None
    ):
        return "https://github.com/paulshuker/coppafisher/raw/HEAD/coppafisher/_version.py"

    return "https://github.com/paulshuker/coppafisher/raw/HEAD/pyproject.toml"


def get_version_from_file(file_lines: list[str]) -> str:
    for file_line in file_lines:
        if not SystemConstants.VERSION_CONTAINS in file_line:
            continue

        return file_line.split(SystemConstants.VERSION_ENCAPSULATE)[1]

    raise ValueError(f"No version found inside file:\n{file_lines}")


def get_software_version() -> str:
    """
    Get coppafisher's version tag as written in _version.py.

    If git CLI is installed, the short form commit hash is appended. This is found by the command `git describe
    --always`. If this fails, then nothing is appended.

    Returns:
        (str): version. The local software version.
    """
    with open(PurePath(os.path.dirname(os.path.realpath(__file__))).parent.joinpath("_version.py"), "r") as f:
        version_tag = get_version_from_file(f.readlines())

    try:
        cwd = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        short_form_commit_hash = (
            subprocess.check_output(["git", "describe", "--always"], cwd=cwd, stderr=subprocess.PIPE).strip().decode()
        )
    except subprocess.CalledProcessError:
        short_form_commit_hash = ""

    if short_form_commit_hash:
        version_tag += "-"
        version_tag += short_form_commit_hash

    return version_tag


def remove_version_hash(version: str) -> str:
    """
    Remove the commit hash code appended to the end of software versions given from the function `get_software_version`.

    Args:
        version (str): the version string.

    Returns
        (str): version_without_hash. The version string without commit hash if there is one.
    """
    return version.split("-")[0]


def get_remote_software_version() -> str | None:
    """
    Get coppafisher's latest version in `_version.py` found online at the default branch.

    Returns:
        (str or none): version_tag. None if the version could not be retrieved.
    """
    fallback = None
    if not web.internet_is_active():
        return fallback

    result = web.try_read_url_at(get_version_url())
    if result is None:
        return fallback
    else:
        return get_version_from_file(result.decode().split("\n"))


def get_available_memory(device: torch.device = None) -> float:
    """
    Get device's available memory at the time of calling this function.

    Args:
        device (torch device): the device. Default: the cpu.

    Returns:
        (float): available_memory. Available memory in GB.
    """
    if device is None:
        device = torch.device("cpu")
    assert type(device) is torch.device

    if device == torch.device("cuda"):
        device_properties = torch.cuda.get_device_properties(device)
        return (device_properties.total_memory - torch.cuda.memory_allocated(device)) / 1e9
    elif device == torch.device("cpu"):
        return psutil.virtual_memory().available / 1e9
    else:
        raise ValueError(f"Unknown device {device}")


def get_device(force_cpu: bool) -> torch.device:
    """
    Get the best device available for pytorch. If not forced to use the CPU and CUDA is available, then the GPU device
    is returned. Otherwise, the CPU is returned.

    Args:
        force_cpu (bool): force return the CPU.

    Returns:
        (`torch.device`): device. Is either torch.device("cpu") or torch.device("cuda").
    """
    if not force_cpu and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def get_core_count() -> int:
    """
    Get the number of CPU cores available for multiprocessing tasks on the system.

    Returns:
        (int): num_cores. The number of available CPU cores.
    """
    n_threads = psutil.cpu_count(logical=True)
    if n_threads is None:
        n_threads = 1
    else:
        n_threads -= 2
    n_threads = np.clip(n_threads, 1, 999, dtype=int)
    return int(n_threads)


def get_terminal_size_xy(x_offset: int = 0, y_offset: int = 0) -> Tuple[int, int]:
    """
    Get the current terminal size in x and y direction, clamped at >= 1 in both directions. Falls back to a default of
    `(80, 20)` if cannot be found.

    Args:
        x_offset (int, optional): add this value to the terminal size in x. Default: 0.
        y_offset (int, optional): add this value to the terminal size in y. Default: 0.

    Returns:
        - (int): number of terminal columns.
        - (int): number of terminal rows.
    """
    terminal_size = tuple(shutil.get_terminal_size((80, 20)))
    return (
        int(np.clip(terminal_size[0] + x_offset, a_min=1, a_max=None)),
        int(np.clip(terminal_size[1] + y_offset, a_min=1, a_max=None)),
    )


def is_path_on_mounted_server(path):
    if sys.platform != "win32":
        # Unix-like systems (Linux, macOS).
        try:
            # Use `df` to check the filesystem type
            result = subprocess.run(["df", "--output=source", path], stdout=subprocess.PIPE, text=True, check=True)
            output = result.stdout.splitlines()[1].strip()  # Get the device name
            # Check if it's a network filesystem (e.g., nfs, cifs, smb)
            if output.startswith(("//", "\\\\")) or any(
                fs_type in output.lower() for fs_type in ["nfs", "cifs", "smb"]
            ):
                return True
            return False
        except subprocess.CalledProcessError:
            return False
    else:
        # Windows-specific check.
        try:
            # Use `wmic` to check if the path is a network drive
            result = subprocess.run(
                ["wmic", "logicaldisk", "where", "drivetype=4", "get", "providername"],
                stdout=subprocess.PIPE,
                text=True,
                check=True,
            )
            # Check if the path is a UNC path or matches a network drive
            if path.startswith(("\\\\", "//")):
                return True
            # Check if the path is on a network drive
            for line in result.stdout.splitlines():
                if not line:
                    continue
                if path.upper().startswith(line.strip().upper()):
                    return True
            return False
        except (subprocess.CalledProcessError, subprocess.FileNotFoundError, FileNotFoundError):
            return False
