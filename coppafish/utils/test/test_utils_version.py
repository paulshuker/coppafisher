from coppafish.utils import system
from coppafish.utils.version import CompatibilityTracker


def test_CompatibilityTracker() -> None:
    tracker = CompatibilityTracker()
    tracker.check("1.0.0", "1.0.1")
    tracker.print_stage_names()

    assert tracker.has_version(
        system.get_software_version()
    ), "Require the latest software version inside of the CompatibilityTracker"