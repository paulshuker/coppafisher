import os


def test_versions_match() -> None:
    # Check that the pyproject version is the same as the version at _version.py.
    version_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "_version.py")

    assert os.path.isfile(version_file_path)

    pyproject_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "pyproject.toml")

    assert os.path.isfile(pyproject_file_path)

    with open(version_file_path, "r") as file:
        version_0 = file.readline().split('"')[1]

    with open(pyproject_file_path, "r") as file:
        pyproject_lines = file.readlines()

    for pyproject_line in pyproject_lines:
        if not pyproject_line.startswith("version = "):
            continue

        version_1 = pyproject_line.split('"')[1]

    assert version_0 == version_1, "pyproject.toml and coppafisher/_version.py must have matching versions"
