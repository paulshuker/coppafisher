import os

from numcodecs import blosc


def set_zarr_global_configs() -> None:
    """
    Set any zarr global configurations before being used to optimise disk reading/writing.
    """
    blosc.use_threads = True
    blosc.set_nthreads(16)


def image_exists(file_path: str) -> bool:
    """
    Checks if an image exists at the given path location.

    Args:
        file_path (str): tile path.

    Returns:
        bool: tile existence.
    """
    # Require a non-empty zarr directory
    return os.path.isdir(file_path) and len(os.listdir(file_path)) > 0
