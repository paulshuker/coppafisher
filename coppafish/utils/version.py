class CompatibilityTracker:
    _page_names = (
        "basic_info",
        "file_names",
        "extract",
        "filter",
        "filter_debug",
        "find_spots",
        "register",
        "register_debug",
        "stitch",
        "ref_spots",
        "call_spots",
        "omp",
    )
    # For each coppafish stage, the value is how to run coppafish, starting again from said stage.
    _stages: dict[str, tuple[str]] = {
        "extract": ("Clear the output directory", "Delete the 'extract' subdirectory inside of the 'tiles' directory"),
        "filter": ("Clear the output directory, including the notebook"),
        "find_spots": (
            "Clear the output directory except the notebook.",
            f"Remove all notebook pages except for {', '.join(_page_names[:5])}",
        ),
        "register": (
            "Clear the output directory except the notebook.",
            f"Remove all notebook pages except for {', '.join(_page_names[:6])}",
        ),
        "stitch": (
            "Clear the output directory except the notebook.",
            f"Remove all notebook pages except for {', '.join(_page_names[:8])}",
        ),
        "ref_spots": (
            "Clear the output directory except the notebook.",
            f"Remove all notebook pages except for {', '.join(_page_names[:9])}",
        ),
        "call_spots": (
            "Clear the output directory except the notebook.",
            f"Remove all notebook pages except for {', '.join(_page_names[:10])}",
        ),
        "omp": (
            "Clear the output directory except the notebook.",
            f"Remove all notebook pages except for {', '.join(_page_names[:11])}",
        ),
        "none": ("Do nothing",),
    }
    # For each coppafish version, the earliest stage is given that requires re-running as a result of the changes.
    _version_compatibility: dict[str, str] = {
        "0.10.7": "none",
        "1.0.0": "extract",
    }

    def __init__(self) -> None:
        pass

    def check(self, from_version: str, to_version: str) -> None:
        """
        Check what output files can be kept when migrating between software versions.
        """
        assert type(from_version) is str
        assert type(to_version) is str
        assert from_version != to_version
        saved_versions_msg = f"valid versions are {', '.join(self._version_compatibility.keys())}"
        if not self.has_version(from_version):
            raise ValueError(f"Could not find version {from_version}, {saved_versions_msg}")
        if not self.has_version(to_version):
            raise ValueError(f"Could not find version {to_version}, {saved_versions_msg}")

        # Find the earliest stage changed from the versions after from_version up to to_version.
        earliest_stage_index = 999
        earliest_stage = "none"
        for i in range(
            list(self._version_compatibility.keys()).index(from_version) + 1,
            list(self._version_compatibility.keys()).index(to_version) + 1,
        ):
            stage = list(self._version_compatibility.values())[i]
            stage_index = list(self._stages.keys()).index(stage)
            if stage_index < earliest_stage_index:
                earliest_stage_index = stage_index
                earliest_stage = list(self._stages.keys())[earliest_stage_index]

        # Find and print the instructions to migrate from the earliest stage.
        print(f"Migrating from coppafish {from_version} to {to_version}:")
        self.print_start_from(earliest_stage)

    def print_start_from(self, stage: str) -> None:
        [print(f"    - {instruction}.") for instruction in self._stages[stage]]

    def print_stage_names(self) -> None:
        stage_names = []
        for stage_name in self._stages.keys():
            if stage_name == "none":
                continue
            stage_names.append(stage_name)
        print(f"Coppafish stages: {', '.join(stage_names)}")

    def has_version(self, version: str) -> bool:
        return version in self._version_compatibility

    def index_of_stage(self, stage: str) -> int:
        assert type(stage) is str

        for i, stage_name in enumerate(self._stages.keys()):
            if stage == stage_name:
                return i
        raise ValueError(f"Failed to find stage name {stage}")
