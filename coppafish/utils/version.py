from collections import OrderedDict


class CompatibilityTracker:
    # For each coppafish stage key, the value is the output files associated with it.
    _stages: OrderedDict[str, tuple[str]] = OrderedDict(
        [
            (
                "extract",
                (
                    "'extract' subdirectory in the tiles directory",
                    "All contents in the output directory, including the notebook",
                ),
            ),
            ("filter", ("'filter' notebook page",)),
            ("find_spots", ("'find_spots' notebook page",)),
            ("register", ("'register' notebook page", "'register_debug' notebook page")),
            ("stitch", ("'stitch' notebook page",)),
            ("reference_spots", ("'ref_spots' notebook page",)),
            ("call_spots", ("'call_spots' notebook page",)),
            (
                "omp",
                (
                    "'omp' notebook page",
                    "results.zgroup in the output directory",
                    "omp_last_config.pkl in the output directory",
                ),
            ),
            ("none", tuple()),
        ]
    )
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

        # Find the earliest stage change from the versions after from_version up to to_version.
        earliest_stage_index = 999
        for i in range(
            list(self._version_compatibility.keys()).index(from_version) + 1,
            list(self._version_compatibility.keys()).index(to_version) + 1,
        ):
            stage = list(self._version_compatibility.values())[i]
            stage_index = list(self._stages.keys()).index(stage)
            if stage_index < earliest_stage_index:
                earliest_stage_index = stage_index

        # Find and print the instructions to migrate from the earliest stage.
        migration_instructions = set()
        for i in range(earliest_stage_index, len(self._stages)):
            stage = list(self._stages.values())[i]
            for instruction in stage:
                migration_instructions.add(instruction)
        print(f"Migrating from coppafish {from_version} to {to_version}, delete the following:")
        [print(f"  - {instruction}") for instruction in migration_instructions]

    def has_version(self, version: str) -> bool:
        return version in self._version_compatibility

    def index_of_stage(self, stage: str) -> int:
        assert type(stage) is str

        for i, stage_name in enumerate(self._stages.keys()):
            if stage == stage_name:
                return i
        raise ValueError(f"Failed to find stage name {stage}")
