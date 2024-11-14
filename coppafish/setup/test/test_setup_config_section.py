from coppafish.setup import config_section


def test_ConfigSection() -> None:
    section = config_section.ConfigSection("debug", {"a": 1, "b": None})

    assert section.name == "debug"
    redundancies = section.list_redundant_params()
    assert type(redundancies) is list
    assert len(redundancies) == 2
    assert "a" in redundancies
    assert "b" in redundancies
    section["a"]
    section["a"]
    section["a"]
    redundancies = section.list_redundant_params()
    assert type(redundancies) is list
    assert len(redundancies) == 1
    assert "b" in redundancies
    assert len(section.items()) == 2
    redundancies = section.list_redundant_params()
    assert type(redundancies) is list
    assert len(redundancies) == 0
