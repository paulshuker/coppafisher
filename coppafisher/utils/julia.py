import os
import subprocess


def check_julia_is_available() -> bool:
    """
    Check if Julia is installed and available via the command line.
    """
    try:
        result = subprocess.run(["julia", "--version"], capture_output=True, text=True, check=True)

        return "julia version 1" in result.stdout.strip().lower()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def run_julia(script_path: str, args: tuple[str]) -> None:
    """
    Run the Julia script given with the given arguments.

    Args:
        script_path (str): the path to the julia file.
        args (tuple of str): arguments appended to the end of the julia call.
    """
    assert type(script_path) is str
    assert type(args) is tuple
    assert os.path.isfile(script_path)

    subprocess.run(["julia", "--optimize", script_path] + list(args), capture_output=False, check=True)
