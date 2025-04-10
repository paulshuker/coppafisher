import os
import tempfile

from coppafisher.utils import system


def test_get_software_version() -> None:
    assert system.get_software_version()
    assert system.get_software_version()[0].isnumeric()
    assert all([num.isnumeric() for num in system.remove_version_hash(system.get_software_version()).split(".")])
    if "-" in system.get_software_version():
        assert len(system.get_software_version().split("-")) == 2


def test_remove_version_hash() -> None:
    assert system.remove_version_hash("v1.0.0") == "v1.0.0"
    assert system.remove_version_hash("v11.0.0") == "v11.0.0"
    assert system.remove_version_hash("v1.2.3") == "v1.2.3"
    assert system.remove_version_hash("v1.2.3-hjdvdvd343egdgesdgs") == "v1.2.3"
    assert system.remove_version_hash("v1.2.3-ds") == "v1.2.3"


def test_is_path_on_mounted_server() -> None:
    tmp_dir = tempfile.TemporaryDirectory("coppafisher")
    tmp_file = os.path.join(tmp_dir.name, "file_naem.png")
    with open(tmp_file, "w") as file:
        file.write("Test")

    assert not system.is_path_on_mounted_server(tmp_dir.name)
    assert not system.is_path_on_mounted_server(tmp_file)
    assert not system.is_path_on_mounted_server(os.path.expanduser("~"))

    tmp_dir.cleanup()
