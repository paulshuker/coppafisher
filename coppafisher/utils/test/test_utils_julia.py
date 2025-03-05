import shutil

from .. import julia


def test_check_julia_is_available() -> None:
    assert julia.check_julia_is_available() == (shutil.which("julia") is not None)
