from coppafish.utils import system
from coppafish.utils.version import CompatibilityTracker


def test_CompatibilityTracker() -> None:
    tracker = CompatibilityTracker()
    tracker.check("0.10.7", "1.0.0")

    assert tracker.has_version(
        system.get_software_version()
    ), "Require the latest software version inside of the CompatibilityTracker"
