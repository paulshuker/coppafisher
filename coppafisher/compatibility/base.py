from collections import OrderedDict

from .. import log
from ..utils import system


class CompatibilityTracker:
    # Every key is a pipeline stage, given in order. Each value is the page names produced during the stage. The
    # pipeline stages are given in chronological order.
    _stages: OrderedDict[str, str] = OrderedDict(
        (
            ("initialisation", "basic_info"),
            ("extract", "extract"),
            ("filter", "filter and filter_debug"),
            ("find_spots", "find_spots"),
            ("register", "register and register_debug"),
            ("stitch", "stitch"),
            ("ref_spots", "ref_spots"),
            ("call_spots", "call_spots"),
            ("omp", "omp"),
            ("none", "none"),
        )
    )
    # For each coppafisher version, this is the earliest stage that requires re-running as a result of the changes is
    # given relative to the version before.
    # NOTE: This must be appended to for each future version release.
    _version_compatibility: OrderedDict[str, str] = OrderedDict(
        (
            ("0.10.6", "extract"),
            ("0.10.7", "filter"),
            ("0.10.8", "none"),
            ("1.0.0", "initialisation"),
            ("1.0.1", "none"),
            ("1.0.2", "none"),
            ("1.0.3", "none"),
            ("1.0.4", "none"),
            ("1.0.5", "none"),
            ("1.0.6", "none"),
            ("1.1.0", "omp"),
            ("1.1.1", "none"),
            ("1.2.0", "filter"),
            ("1.2.1", "none"),
            ("1.2.2", "register"),
            ("1.2.3", "call_spots"),
            ("1.2.4", "none"),
            ("1.2.5", "none"),
            ("1.2.6", "none"),
            ("1.2.7", "none"),
            ("1.3.0", "call_spots"),
            ("1.4.0", "none"),
            ("1.4.1", "none"),
            ("1.4.2", "none"),
            ("1.5.0", "none"),
        )
    )
    _stage_instructions: list[tuple[str, ...]]

    def __init__(self) -> None:
        # For each stage, instructions are given on how to remove all data during and after said stage.
        self._stage_instructions = [
            (
                "Clear the output directory. Delete the notebook",
                "Delete the 'extract' subdirectory inside of the 'tiles' directory",
            ),
            (
                "Clear the output directory. Delete the notebook",
                "Delete the 'extract' subdirectory inside of the 'tiles' directory",
            ),
            ("Clear the output directory. Delete the notebook",),
            (
                "Clear the output directory except the notebook",
                f"Remove notebook page {list(self._stages.values())[3]} and all later pages",
            ),
            (
                "Clear the output directory except the notebook",
                f"Remove notebook page {list(self._stages.values())[4]} and all later pages",
            ),
            (
                "Clear the output directory except the notebook",
                f"Remove notebook page {list(self._stages.values())[5]} and all later pages",
            ),
            (
                "Clear the output directory except the notebook",
                f"Remove notebook page {list(self._stages.values())[6]} and all later pages",
            ),
            (
                "Clear the output directory except the notebook",
                f"Remove notebook page {list(self._stages.values())[7]} and all later pages",
            ),
            (
                "Clear the output directory except the notebook",
                f"Remove notebook page {list(self._stages.values())[8]} and all later pages",
            ),
            ("Do nothing",),
        ]
        assert len(self._stages) == len(self._stage_instructions)

    def check(self, from_version: str, to_version: str) -> tuple[str, ...]:
        """
        Check what output files can be kept when migrating between software versions.

        Args:
            from_version (str): old software version.
            to_version (str): software version migrating to.

        Returns:
            (tuple of str): output. Each given line of output.
        """
        assert type(from_version) is str
        assert type(to_version) is str
        assert from_version != to_version

        from_version = system.remove_version_hash(from_version)
        to_version = system.remove_version_hash(to_version)
        saved_versions_msg = f"valid versions are {', '.join(self._version_compatibility.keys())}"
        if not self.has_version(from_version):
            raise ValueError(f"Could not find version {from_version}, {saved_versions_msg}")
        if not self.has_version(to_version):
            raise ValueError(f"Could not find version {to_version}, {saved_versions_msg}")

        # Find the earliest stage changed from the versions after from_version up to to_version.
        earliest_stage, _ = self._get_earliest_stage_between(from_version, to_version)

        # Find and print the instructions to migrate from the earliest stage.
        instructions = []
        instructions.append(f"Migrating from coppafisher {from_version} to {to_version}:")
        instructions += self.get_start_from(earliest_stage)
        instructions.append(
            "To delete notebook pages, see Usage -> Advanced Usage at "
            + "https://paulshuker.github.io/coppafisher/advanced_usage/#delete-notebook-page in the documentation"
        )
        for instruction in instructions:
            log.info(instruction)

        return instructions

    def is_notebook_compatible(self, nb_page_versions: dict[str, str], current_version: str | None = None) -> bool:
        """
        Check if the notebook contains incompatible data from older software versions. If so, a warning is printed and
        false is returned.

        Args:
            nb_page_versions (dict[str, str]): every notebook page name as a key, each value is the notebook page's
                software version when it was created.
            current_version (str, optional): this current software version. Default: value in coppafisher/_version.py.

        Returns:
            (bool): valid. Whether all the notebook's data is compatible for this version of coppafisher.
        """
        assert type(nb_page_versions) is dict
        if current_version is None:
            current_version = system.get_software_version()
        assert type(current_version) is str
        current_version = system.remove_version_hash(current_version)
        assert current_version in self._version_compatibility, f"Unknown version {current_version} given"

        for page_name, page_version in nb_page_versions.items():
            page_version = system.remove_version_hash(page_version)
            _, page_stage_index = self._get_stage_with_page_name(page_name)
            if page_version not in self._version_compatibility:
                raise ValueError(f"Notebook page {page_name} has unknown software version: {page_version}")
            # For a page and its version, if the earliest stage page that must be removed from the notebook for current
            # compatibility is equal to or earlier than this page, then the notebook is invalid.
            earliest_stage, earliest_stage_index = self._get_earliest_stage_between(page_version, current_version)
            if earliest_stage_index <= page_stage_index:
                # The notebook has backwards incompatibilities.
                log.warn(
                    f"The existing notebook contains backwards incompatibility on page {page_name} at version "
                    + f"{page_version} compared to current version {current_version}."
                )
                log.warn("The suggested course of action is:")
                [log.info(instruction) for instruction in self.get_start_from(earliest_stage)]

                return False

        return not self._notebook_has_downgrade(nb_page_versions, current_version)

    def get_start_from(self, stage: str) -> tuple[str, ...]:
        """
        Print the instructions on how to prepare the coppafisher pipeline to start from the given stage.

        Args:
            stage (str): the stage to start again from.

        Returns:
            (tuple of str): each instruction given.
        """
        instructions = self._stage_instructions[self._get_stage_index(stage)]

        return instructions

    def print_stage_names(self) -> str:
        """
        Print every stage that is part of coppafisher's pipeline in chronological order.

        Returns:
            (str): message. The message printed.
        """
        stage_names = []
        for stage_name in self._stages:
            if stage_name == "none":
                continue
            stage_names.append(stage_name)
        message = f"Coppafisher stages: {', '.join(stage_names)}"
        print(message)

        return message

    def get_page_names_added_after(self, page_name: str) -> tuple[str, ...]:
        """
        Get every page name added after the given page name during the usual pipeline chronological order.

        If the given page name's stage has two or more page names added during it, these other page names are also
        returned.

        Args:
            page_name (str): the page name.

        Returns:
            (tuple of str): page_names_after.
        """
        _, stage_index = self._get_stage_with_page_name(page_name)
        page_names_after = []
        for page_names_str in list(self._stages.values())[stage_index:-1]:
            for page_name_after in self._parse_page_names(page_names_str):
                if page_name_after == page_name:
                    continue
                page_names_after.append(page_name_after)

        return tuple(page_names_after)

    def has_version(self, version: str) -> bool:
        """
        Check if a version exists in the tracker's records.

        Args:
            version (str): the version to look for.

        Returns:
            (bool): version_exists. Whether the version exists.
        """
        return system.remove_version_hash(version) in self._version_compatibility

    def _notebook_has_downgrade(self, nb_page_versions: dict[str, str], current_version: str) -> bool:
        """
        Check the given notebook for a version drop when going through the stages in order, which should be impossible.
        We do not support users reverting back versions of the software and continuing a pipeline run as this has
        unpredictable consequences.

        Args:
            nb_page_versions (dict[str, str]): every notebook page name as a key, each value is the notebook page's
                software version when it was created.
            current_version (str): this current software version.

        Returns:
            (bool) has_downgrade: true if the software version did downgrade.
        """
        current_version = system.remove_version_hash(current_version)
        ordered_version_list = list(self._version_compatibility)
        current_version_index = ordered_version_list.index(current_version)
        for page_name, page_version in nb_page_versions.items():
            page_version = system.remove_version_hash(page_version)
            page_version_index = ordered_version_list.index(page_version)
            if page_version_index > current_version_index:
                log.warn(
                    f"Notebook contains page {page_name} run on version {page_version} which is higher than the "
                    + f"current coppafisher version ({current_version})."
                )
                log.warn("The suggested course of action is:")
                log.warn(f"    - Update coppafisher version to >= {page_version} before re-running.")
                return True
        return False

    def _get_earliest_stage_between(self, from_version_exclusive: str, to_version_inclusive: str) -> tuple[str, int]:
        index_start = list(self._version_compatibility.keys()).index(from_version_exclusive) + 1
        index_end = list(self._version_compatibility.keys()).index(to_version_inclusive) + 1
        earliest_stage = "none"
        earliest_stage_index = 999
        for index in range(index_start, index_end):
            stage = list(self._version_compatibility.values())[index]
            index = self._get_stage_index(stage)
            if index < earliest_stage_index:
                earliest_stage = stage
                earliest_stage_index = index
        return earliest_stage, earliest_stage_index

    def _get_stage_index(self, stage: str) -> int:
        for i, s in enumerate(self._stages):
            if s == stage:
                return i
        raise ValueError(f"Failed to find a stage called {stage}")

    def _get_stage_with_page_name(self, page_name: str) -> tuple[str, int]:
        for i, (stage_name, page_names) in enumerate(self._stages.items()):
            if page_name in self._parse_page_names(page_names):
                return stage_name, i

    def _parse_page_names(self, page_name_str: str) -> list[str]:
        return page_name_str.split(" and ")
