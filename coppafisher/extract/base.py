import os

import tqdm

from ..pipeline import extract_run
from ..setup.config import Config
from ..utils import system, zarray


def update_tile_dir(config_path: str) -> None:
    """
    Migrate extract's tile directory for the latest coppafisher version.

    The data is converted at `tile_dir` under `[file_names]` in the config from unzipped zarr.DirectoryStores into
    zarr.ZipStores.

    Args:
        config_path (str): the path to the config.
    """
    if not isinstance(config_path, str):
        raise TypeError(f"Unexpected config_path type: {type(config_path)}")

    config_path = str(config_path)
    config = Config()
    config.load(config_path, post_check=False)

    tile_dir = config["file_names"]["tile_dir"]
    extract_dir = os.path.join(tile_dir, "extract")
    if tile_dir is None:
        raise ValueError(f"tile_dir must be specified under the [file_names] section in {config_path}")
    if not os.path.isdir(tile_dir):
        raise SystemError(f"Could not find tile directory at {tile_dir}")
    if not os.path.isdir(extract_dir):
        raise SystemError(f"Could not find extract directory at {extract_dir}")
    if not os.listdir(extract_dir):
        raise FileNotFoundError(f"No files found inside {extract_dir}")

    file_names = []
    for file_name in os.listdir(extract_dir):
        file_path = os.path.join(extract_dir, file_name)
        if os.path.isfile(file_path):
            continue
        file_names.append(file_name)

    if not file_names:
        print("Nothing to update")
        return

    version_path = os.path.join(extract_dir, extract_run.VERSION_FILE_NAME)
    if not os.path.isfile(version_path):
        raise FileNotFoundError(f"Could not find version file at {version_path}")
    with open(version_path, "w") as file:
        file.write(system.get_software_version())

    for file_name in tqdm.tqdm(file_names, desc="Migrating extract files"):
        file_path = os.path.join(extract_dir, file_name)
        zarray.convert_array_to_zip_store(file_path)
