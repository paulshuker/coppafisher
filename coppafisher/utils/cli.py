import subprocess


def has_cli_tool(name: str) -> bool:
    """
    Checks if a command line interface with the given name exists.

    Args:
        name (str): the name of the CLI tool to check for.

    Returns
        (bool) does_exist: whether the tool exists.
    """
    assert name

    try:
        subprocess.run([name, "-h"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
