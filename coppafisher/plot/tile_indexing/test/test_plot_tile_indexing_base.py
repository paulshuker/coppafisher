from os import path

import pytest

from .. import base


@pytest.mark.notebook
def test_view_tile_indexing_grid() -> None:
    config_file_path = path.dirname(path.dirname(path.dirname(path.dirname(__file__))))
    config_file_path = path.join(config_file_path, "robominnie", "test", ".integration_dir", "robominnie.ini")
    assert path.isfile(config_file_path)

    try:
        base.view_tile_indexing_grid("/non-existing/path/to/config.ini", show=False)
        raise AssertionError("Expected FileNotFoundError to be raised")
    except FileNotFoundError:
        pass

    base.view_tile_indexing_grid(config_file_path, show=False)
