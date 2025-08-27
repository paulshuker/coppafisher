import json
import os
import shutil
import time
from typing import Any, Optional, Tuple

import numpy as np

from .. import log
from ..compatibility import CompatibilityTracker
from ..utils import system as utils_system
from .config import Config
from .notebook_page import NotebookPage


class Notebook:
    _directory: str

    def get_directory(self) -> str:
        return self._directory

    # The notebook's full filepath.
    # NOTE: This is not fixed as users can move the notebook's location. Therefore, use with caution.
    directory: str = property(get_directory)

    # Attribute names allowed to be set inside the notebook page that are not in _options.
    _valid_attribute_names = (
        "config_path",
        "_config_path",
        "_init_config",
        "_directory",
        "_time_created",
        "_version",
    )
    _debug_pages = ("debug", "debug_2")

    _config_path: Optional[str]

    def get_config_path(self) -> Optional[str]:
        return self._config_path

    # The latest config file path used by coppafisher to run the pipeline.
    # There is no guarantee that the config file is still there when an old notebook is loaded in.
    config_path: Optional[str] = property(get_config_path)

    _metadata_name = "_metadata.json"

    _time_created: float
    _time_created_key = "time_created"
    _version: str
    _version_key = "version"

    _options = {
        "basic_info": [
            "*basic_info* page contains information that is used at all stages of the pipeline.",
        ],
        "extract": [
            "*extract* page contains information related to extraction of raw input files for use in coppafisher.",
        ],
        "filter": [
            "*filter* page contains information on image filtering applied to extracted images.",
        ],
        "filter_debug": [
            "*filter_debug* page contains additional information on filtering that is not used later in the pipeline.",
        ],
        "find_spots": [
            "*find_spots* page contains information about spots found on every tiles, rounds and channels.",
        ],
        "stitch": [
            "*stitch* page contains information about how tiles were stitched together to produce global coords.",
        ],
        "register": [
            "*register* page contains best found solutions to allign images.",
        ],
        "register_debug": [
            "*register_debug* page contains information on how the image allignments in *register* were calculated.",
        ],
        "ref_spots": [
            "*ref_spots* page contains gene assignments and info for spots found on reference round.",
        ],
        "call_spots": [
            "*call_spots* page contains `bleed_matrix` and expected code for each gene.",
        ],
        "omp": [
            "*omp* page contains gene assigments and information for spots found through Orthogonal Matching Pursuit.",
        ],
        "thresholds": [
            "*thresholds* page contains quality thresholds which affect which spots plotted and exported to pciSeq.",
        ],
        "debug": [
            "*debug* page for unit testing.",
        ],
        "debug_2": [
            "*debug* page for unit testing.",
        ],
    }

    def __init__(self, notebook_dir: str, config_path: Optional[str] = None, must_exist: bool = True) -> None:
        """
        Load the notebook found at the given directory. Or, if the directory does not exist, create the directory.

        Args:
            notebook_dir (str): the notebook directory to write into and/or load from.
            config_path (str, optional): path to the pipeline's config file. This must be given for new pages to be
                added, i.e. during the pipeline runtime. Default: not given.
            must_exists (bool, optional): crash if the notebook does not already exist. Default: true.
        """
        assert type(notebook_dir) is str
        assert config_path is None or type(config_path) is str
        if must_exist and not os.path.isdir(notebook_dir):
            raise FileNotFoundError(f"No notebook at {notebook_dir}")

        self._config_path = None
        if config_path is not None:
            self._config_path = os.path.abspath(config_path)
        self._directory = os.path.abspath(notebook_dir)
        self._time_created = time.time()
        self._version = utils_system.get_software_version()
        if not os.path.isdir(self._directory):
            if self._config_path is None:
                raise ValueError("To create a new notebook, config_path must be specified")
            log.info(f"Creating notebook at {self._directory}")
            os.mkdir(self._directory)
            self._save()
        self._load()

    def __iadd__(self, page: NotebookPage):
        """
        Add and save a new page to the notebook using syntax notebook += notebook_page.
        """
        if type(page) is not NotebookPage:
            raise TypeError(f"Cannot add type {type(page)} to the notebook")
        if self._config_path is None:
            raise ValueError("The notebook must have a specified config_path when instantiated to add notebook pages.")
        if not os.path.isfile(self._config_path) and page.name not in self._debug_pages:
            raise FileNotFoundError(f"Could not add page since config at {self._config_path} was not found")
        unset_variables = page.get_unset_variables()
        if len(unset_variables) > 0:
            raise ValueError(
                f"Page {page.name} must have every variable set before adding to the notebook. "
                + f"Variable(s) unset: {', '.join(unset_variables)}"
            )

        if page.name not in self._debug_pages and len(self._get_modified_config_variables()) > 0:
            log.warn(
                f"The config at {self.config_path} has modified variable(s): "
                + ", ".join(self._get_modified_config_variables())
                + " since the pipeline was first started. Continue at your own risk."
            )
        self.__setattr__(page.name, page)
        self._save()
        return self

    def has_page(self, page_name: str) -> bool:
        assert type(page_name) is str
        if page_name not in self._options.keys():
            raise ValueError(f"Not a real page name: {page_name}. Expected one of {', '.join(self._options.keys())}")

        try:
            self.__getattribute__(page_name)
            return True
        except AttributeError:
            return False

    def delete_page(self, page_name: str, prompt: bool = True) -> None:
        """
        Delete a notebook page from disk and memory. This cannot be undone!

        Args:
            page_name (str): page name to delete.
            prompt (bool, optional): give the user a y/n prompt for other suggestions to the notebook. Default: true.

        Notes:
            - This function is helpful for users when they have finished version compatibility checks using the
                CompatibilityTracker and now wish to delete incompatible notebook pages.
        """
        assert type(page_name) is str
        if not self.has_page(page_name):
            raise ValueError(f"Page name {page_name} not found")
        page_names_after = [] if page_name.startswith("debug") else self._get_page_names_after_page(page_name)
        if prompt and len(page_names_after) > 0:
            print(f"The notebook contains pages {', '.join(page_names_after)} that were added after page {page_name}.")
            result = input("Do you want to delete these pages too (recommended)? (y/n): ")
            if result == "y":
                for earlier_page_name in page_names_after:
                    self.delete_page(earlier_page_name, prompt=False)
        page_name_directory = self._get_page_directory(page_name)
        page: NotebookPage = self.__getattribute__(page_name)
        self.__delattr__(page_name)
        page.close_stores()
        shutil.rmtree(page_name_directory)
        print(f"{page_name} deleted")

    def resave(self) -> None:
        """
        Delete the notebook on disk and re-save every page using the instance in memory.
        """
        # NOTE: This function should not be used by the coppafisher pipeline. This is purely a function for developers
        # to manually change variables that are already saved to disk. Even then, this function should be used as
        # little as possible as it will inevitably cause bugs.
        start_time = time.time()
        for filename in os.listdir(self._directory):
            filepath = os.path.join(self._directory, filename)
            if os.path.isfile(filepath) and filepath != self._get_metadata_path():
                raise SystemError(f"Unexpected file called {filename} found in {self._directory}")
            if os.path.isdir(filepath) and filename not in self._options:
                raise SystemError(f"Unexpected directory called {filename} found in {self._directory}")
            if os.path.isdir(filepath):
                if filename in self._get_added_page_names():
                    self.__getattribute__(filename).resave(filepath)
                else:
                    shutil.rmtree(filepath)
        os.remove(self._get_metadata_path())
        self._save_metadata()
        end_time = time.time()
        print(f"Notebook re-saved in {end_time - start_time:.2f}s")

    def get_all_versions(self) -> dict[str, str]:
        """
        Get every page's software version.

        Returns:
            (dict[str, str]) all_versions: each key is a page name, the value is its software version.
        """
        all_versions = {}
        for page in self._get_added_pages():
            all_versions[page.name] = page.version
        return all_versions

    def zip(self) -> None:
        """
        Zip all notebook page zarr Array/Group variables.

        Does nothing if they are already zipped.
        """
        if all([not page.get_unzipped_variables() for page in self._get_added_pages()]):
            print("Nothing to zip")
            return
        for page in self._get_added_pages():
            page.zip(self._get_page_directory(page.name))

    def __setattr__(self, name: str, value: Any, /) -> None:
        """
        Deals with syntax `notebook.name = value`.
        """
        if name in self._valid_attribute_names:
            object.__setattr__(self, name, value)
            return

        if type(value) is not NotebookPage:
            raise TypeError(f"Can only add NotebookPage classes to the Notebook, got {type(value)}")
        if self.has_page(value.name):
            raise ValueError(f"Notebook already contains page named {value.name}")

        object.__setattr__(self, name, value)

    def __gt__(self, page_name: str) -> None:
        """
        Print a page's description by doing `notebook > "page_name"`.
        """
        assert type(page_name) is str
        if page_name not in self._options.keys():
            print(f"No page named {page_name}")
            return

        print(f"Page name {page_name}:")
        print(f"\tVariable count: {NotebookPage(page_name).get_variable_count()}")
        print(f"\tDescription: {self._options[page_name][0]}")

    def __del__(self) -> None:
        for page in self._get_added_pages():
            page.close_stores()

    def _save(self) -> None:
        """
        Save the notebook to the directory specified when the notebook was instantiated.
        """
        start_time = time.time()
        self._save_metadata()
        for page in self._get_added_pages():
            page_dir = self._get_page_directory(page.name)
            page.save(page_dir)
        end_time = time.time()
        log.info(f"Notebook saved in {end_time - start_time:.2f}s")

    def _load(self) -> None:
        self._load_metadata()
        # Check directory for existing notebook pages and load them in.
        for page_name in os.listdir(self._directory):
            if page_name == self._metadata_name:
                continue
            page_path = self._get_page_directory(page_name)
            if os.path.isfile(page_path):
                raise FileExistsError(f"Unexpected file {page_path} inside the notebook")
            if page_name not in self._options.keys():
                raise IsADirectoryError(f"Unexpected directory at {page_path} inside the notebook")
            loaded_page = NotebookPage(page_name)
            loaded_page.load(page_path)
            self.__setattr__(page_name, loaded_page)

    def _get_added_pages(self) -> Tuple[NotebookPage, ...]:
        pages = []
        for page_name in self._options.keys():
            if self.has_page(page_name):
                pages.append(self.__getattribute__(page_name))
        return tuple(pages)

    def _get_added_page_names(self) -> Tuple[str, ...]:
        pages = self._get_added_pages()
        names = []
        for page in pages:
            names.append(page.name)
        return tuple(names)

    def _get_page_directory(self, page_name: str) -> str:
        assert type(page_name) is str

        return str(os.path.join(self._directory, page_name))

    def _get_modified_config_variables(self) -> Tuple[str, ...]:
        assert self.config_path is not None

        modified_variables = tuple()
        msg_prefix = f"Config at {self.config_path} is missing"
        msg_suffix = (
            "Is the notebook from a different software version? If you are unsure, it "
            + "is recommended to delete the notebook and re-run the pipeline."
        )
        config_on_disk = Config()
        config_on_disk.load(self.config_path, post_check=False)

        for page in self._get_added_pages():
            for config_section in page.associated_configs:
                if config_section not in config_on_disk.get_section_names():
                    log.warn(f"{msg_prefix} section {config_section}. {msg_suffix}")
                    continue
                for var_name, value in page.associated_configs[config_section].items():
                    is_equal = False
                    if var_name not in config_on_disk[config_section].get_parameter_names():
                        log.warn(f"{msg_prefix} variable named {var_name} in section {config_section}. {msg_suffix}")
                        modified_variables += (var_name,)
                        continue
                    config_variable = config_on_disk[config_section][var_name]
                    if value == config_variable:
                        is_equal = True
                    if type(value) is list:
                        array_0 = np.array(value)
                        array_1 = np.array(config_variable)
                        # This is dumb. But, it works.
                        if isinstance(array_0.dtype.type(), (str, np.str_)):
                            is_equal = (array_0 == array_1).all()
                        else:
                            is_equal = np.allclose(array_0, array_1)
                    if not is_equal:
                        modified_variables += (var_name,)
        return modified_variables

    def _save_metadata(self) -> None:
        assert os.path.isdir(self._directory)

        file_path = self._get_metadata_path()
        if os.path.isfile(file_path):
            return
        metadata = {
            self._time_created_key: self._time_created,
            self._version_key: self._version,
        }
        with open(file_path, "x") as file:
            file.write(json.dumps(metadata, indent=4))

    def _load_metadata(self) -> None:
        assert os.path.isdir(self._directory)
        file_path = self._get_metadata_path()
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Could not find notebook metadata at {file_path}")

        metadata = dict()
        with open(file_path, "r") as file:
            metadata = json.loads(file.read())
        self._version = metadata[self._version_key]
        self._time_created = metadata[self._time_created_key]

    def _get_metadata_path(self) -> str:
        return os.path.join(self._directory, self._metadata_name)

    def _get_page_names_after_page(self, page_name: str) -> tuple[str, ...]:
        """
        Get all the existing pages that are run on a pipeline stage after the given page name.

        Args:
            page_name (str): the page name to consider.

        Returns:
            (tuple of str): page_names_after. A sequence of page names that run after the given page name in the
                pipeline's chronological order.
        """
        assert self.has_page(page_name)

        tracker = CompatibilityTracker()
        all_page_names_after = tracker.get_page_names_added_after(page_name)
        page_names_after = []

        page: NotebookPage = self.__getattribute__(page_name)
        for other_page in self._get_added_pages():
            if other_page == page:
                continue
            if other_page.name not in all_page_names_after:
                continue
            page_names_after.append(other_page.name)
        return tuple(page_names_after)
