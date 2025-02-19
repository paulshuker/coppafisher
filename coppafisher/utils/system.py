import os
import shutil
import socket
import ssl
import urllib
from pathlib import PurePath
from typing import Tuple

import numpy as np
import psutil
import torch


class SystemConstants:
    VERSION_URL = "https://github.com/paulshuker/coppafisher/raw/HEAD/coppafisher/_version.py"
    # The character(s) that encapsulate the software version tag in _version.py, in this case it is quotation marks.
    VERSION_ENCAPSULATE = '"'


def get_software_version() -> str:
    """
    Get coppafisher's version tag written in _version.py

    Returns:
        str: software version.
    """
    consts = SystemConstants()
    with open(PurePath(os.path.dirname(os.path.realpath(__file__))).parent.joinpath("_version.py"), "r") as f:
        version_tag = f.read().split(consts.VERSION_ENCAPSULATE)[1]
    return version_tag


def get_remote_software_version() -> str:
    """
    Get coppafisher's latest version in `_version.py` found online at the default branch.

    Returns:
        str: version tag. None if the version could not be retrieved.
    """
    consts = SystemConstants()
    fallback = None
    if not internet_is_active():
        return fallback
    try:
        f = urllib.request.urlopen(consts.VERSION_URL)
        version_contents = str(f.read())
        index_start = version_contents.index(consts.VERSION_ENCAPSULATE)
        index_end = version_contents.index(consts.VERSION_ENCAPSULATE, index_start + 1)
    except (urllib.error.HTTPError, urllib.error.URLError):
        # This can be reached if GitHub refuses the request due to too many recent requests.
        return fallback
    return version_contents[index_start + 1 : index_end]


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


def internet_is_active() -> bool:
    """
    Check for an internet connection.

    Returns:
        bool: whether the system is connected to the internet.
    """
    try:
        urllib.request.urlopen("http://www.google.com")
        return True
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        ValueError,
        socket.gaierror,
        TimeoutError,
        OSError,
        ssl.SSLError,
        ConnectionResetError,
        FileNotFoundError,
    ):
        return False
