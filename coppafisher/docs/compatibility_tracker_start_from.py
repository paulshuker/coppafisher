from coppafisher.compatibility import CompatibilityTracker

tracker = CompatibilityTracker()
print("\n".join(tracker.get_start_from("find_spots")))
