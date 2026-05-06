from coppafisher.utils import cli


def test_has_cli_tool() -> None:
    assert cli.has_cli_tool("7z")
    assert not cli.has_cli_tool("efkjheghejndvnsdfeklfjekf")
