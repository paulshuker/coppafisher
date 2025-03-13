from collections import OrderedDict

from coppafisher.compatibility import CompatibilityTracker
from coppafisher.utils import system


def test_CompatibilityTracker() -> None:
    tracker = CompatibilityTracker()
    tracker.check("0.10.7", "1.0.0")
    tracker.print_stage_names()
    assert tracker.has_version(
        system.get_software_version()
    ), "Require the latest software version inside of the coppafisher/compatibility/base.py"
    assert not tracker.has_version("abc")

    tracker._stages = OrderedDict(
        [("stage_0", "page_0 and page_1"), ("stage_1", "page_2"), ("stage_2", "page_3 and page_4"), ("none", "none")]
    )
    tracker._version_compatibility = OrderedDict(
        [
            ("0.1.0", "stage_0"),
            ("0.2.0", "stage_1"),
            ("0.3.0", "stage_0"),
            ("0.4.0", "stage_2"),
            ("0.5.0", "stage_2"),
            ("0.7.0", "stage_1"),
            ("1.0.0", "stage_0"),
        ]
    )
    tracker._stage_instructions = [
        ("Starting from stage 0 instructions",),
        ("Starting from stage 1 instructions",),
        ("Starting from stage 2 instructions",),
    ]

    assert "stage_0, stage_1, stage_2" in tracker.print_stage_names()

    assert sorted(list(tracker.get_page_names_added_after("page_0"))) == ["page_1", "page_2", "page_3", "page_4"]
    assert sorted(list(tracker.get_page_names_added_after("page_1"))) == ["page_0", "page_2", "page_3", "page_4"]

    assert tracker.has_version("0.1.0")
    assert tracker.has_version("0.2.0")
    assert tracker.has_version("0.3.0")
    assert tracker.has_version("0.4.0")
    assert tracker.has_version("0.5.0")
    assert tracker.has_version("0.7.0")
    assert tracker.has_version("1.0.0")

    assert not tracker.has_version("0.1.0x")
    assert not tracker.has_version("1.2.0")
    assert not tracker.has_version("2.3.0")
    assert not tracker.has_version("5.4.0")
    assert not tracker.has_version("0.0.0")
    assert not tracker.has_version("0.a.0")
    assert not tracker.has_version("0")
    assert not tracker.has_version("1")

    output = tracker.check("0.1.0", "0.2.0")
    assert tracker._stage_instructions[0][0] not in output
    assert tracker._stage_instructions[1][0] in output
    assert tracker._stage_instructions[2][0] not in output

    output = tracker.check("0.2.0", "0.3.0")
    assert tracker._stage_instructions[0][0] in output
    assert tracker._stage_instructions[1][0] not in output
    assert tracker._stage_instructions[2][0] not in output

    output = tracker.check("0.1.0", "0.3.0")
    assert tracker._stage_instructions[0][0] in output
    assert tracker._stage_instructions[1][0] not in output
    assert tracker._stage_instructions[2][0] not in output

    output = tracker.check("0.3.0", "0.4.0")
    assert tracker._stage_instructions[0][0] not in output
    assert tracker._stage_instructions[1][0] not in output
    assert tracker._stage_instructions[2][0] in output

    nb_page_versions = {"page_0": "0.4.0", "page_1": "0.4.0"}
    current_version = "0.4.0"
    assert tracker.is_notebook_compatible(nb_page_versions=nb_page_versions, current_version=current_version)
    nb_page_versions = {"page_0": "0.4.0", "page_1": "0.4.0", "page_2": "0.4.0"}
    current_version = "0.4.0"
    assert tracker.is_notebook_compatible(nb_page_versions=nb_page_versions, current_version=current_version)
    nb_page_versions = {"page_0": "0.4.0", "page_1": "0.4.0", "page_2": "0.4.0", "page_3": "0.4.0"}
    current_version = "0.4.0"
    assert tracker.is_notebook_compatible(nb_page_versions=nb_page_versions, current_version=current_version)
    nb_page_versions = {"page_0": "0.4.0", "page_1": "0.4.0", "page_2": "0.4.0", "page_3": "0.2.0"}
    current_version = "0.4.0"
    assert not tracker.is_notebook_compatible(nb_page_versions=nb_page_versions, current_version=current_version)
    nb_page_versions = {"page_0": "0.3.0", "page_1": "0.4.0", "page_2": "0.4.0", "page_3": "0.4.0"}
    current_version = "0.4.0"
    assert tracker.is_notebook_compatible(nb_page_versions=nb_page_versions, current_version=current_version)
    nb_page_versions = {"page_0": "0.4.0", "page_1": "0.3.0", "page_2": "0.4.0", "page_3": "0.4.0"}
    current_version = "0.4.0"
    assert tracker.is_notebook_compatible(nb_page_versions=nb_page_versions, current_version=current_version)
    nb_page_versions = {"page_0": "0.4.0", "page_1": "0.4.0", "page_2": "0.3.0", "page_3": "0.4.0"}
    current_version = "0.4.0"
    assert tracker.is_notebook_compatible(nb_page_versions=nb_page_versions, current_version=current_version)
    nb_page_versions = {"page_0": "0.4.0", "page_1": "0.4.0", "page_2": "0.4.0", "page_3": "0.3.0"}
    current_version = "0.4.0"
    assert not tracker.is_notebook_compatible(nb_page_versions=nb_page_versions, current_version=current_version)
    nb_page_versions = {"page_0": "0.4.0", "page_1": "0.4.0", "page_2": "0.4.0", "page_3": "0.4.0", "page_4": "0.3.0"}
    current_version = "0.4.0"
    assert not tracker.is_notebook_compatible(nb_page_versions=nb_page_versions, current_version=current_version)

    current_version = "0.5.0"
    nb_page_versions_base = {
        "page_0": "0.5.0",
        "page_1": "0.5.0",
        "page_2": "0.5.0",
        "page_3": "0.5.0",
        "page_4": "0.5.0",
    }
    for page_number in range(3):
        nb_page_versions = nb_page_versions_base.copy()
        for i in range(page_number, 3):
            nb_page_versions[f"page_{i}"] = "0.4.0"
        assert tracker.is_notebook_compatible(nb_page_versions=nb_page_versions, current_version=current_version)
        assert not tracker._notebook_has_downgrade(nb_page_versions, current_version)

    assert tracker._notebook_has_downgrade({"page_0": "1.0.0", "page_1": "0.5.0"}, "0.4.0")
    assert tracker._notebook_has_downgrade({"page_0": "1.0.0", "page_1": "0.5.0"}, "0.5.0")
    assert tracker._notebook_has_downgrade({"page_0": "1.0.0", "page_1": "0.5.0"}, "0.7.0")
    assert not tracker._notebook_has_downgrade({"page_0": "1.0.0", "page_1": "0.5.0"}, "1.0.0")
