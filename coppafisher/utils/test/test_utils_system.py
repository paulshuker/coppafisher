import os
import tempfile

from coppafisher.utils import system


def test_is_path_on_mounted_server() -> None:
    tmp_dir = tempfile.TemporaryDirectory("coppafisher")
    tmp_file = os.path.join(tmp_dir.name, "file_naem.png")
    with open(tmp_file, "w") as file:
        file.write("Test")

    assert not system.is_path_on_mounted_server(tmp_dir.name)
    assert not system.is_path_on_mounted_server(tmp_file)
    assert not system.is_path_on_mounted_server(os.path.expanduser("~"))

    tmp_dir.cleanup()
