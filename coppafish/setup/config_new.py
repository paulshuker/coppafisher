from collections.abc import Callable
import configparser
import importlib.resources as importlib_resources
import os
from os import path
import re
from typing import Any

from .config_section import ConfigSection
from .. import log


class Config:
    """
    Reads and validates config files.

    Every section is separated into ConfigSection classes with parameters that are accessed like a dictionary. For
    example, you can gather the variable "use_tiles" from basic_info by

    ```py
    use_tiles = config["basic_info"]["use_tiles"]
    ```

    (This allows for legacy code support as the config used to be a plain python dictionary object).
    """

    _sections: list[ConfigSection]

    # Parameter pre checkers cannot be combined except for the "maybe" keyword, separated by an underscore. The check
    # must be true for the parameter to be valid. These checkers are applied BEFORE the parameter is formatted into its
    # correct type, so the parameter is still a string.
    _param_pre_checks: dict[str, Callable[[str], bool]] = {
        "maybe": lambda x: x == "",
        "int": lambda x: re.match("-?[0-9]+", x) is not None,
        "number": lambda x: re.match(r"-?[0-9]+(\.[0-9]+)?$", x) is not None,
        "bool": lambda x: re.match("true|false", x, re.IGNORECASE) is not None,
        "str": lambda x: len(x) > 0,
        "file": lambda x: len(x) > 0,
        "dir": lambda x: len(x) > 0,
        "tuple": lambda _: True,
    }

    # Parameter formatters are callables that convert the given parameter config string value into said format.
    _param_formatters: dict[str, Callable[[str], None | int | float | bool | str | tuple[Any, ...]]] = {
        "maybe": lambda x: None if x == "" else x,
        "int": lambda x: int(x),
        "number": lambda x: float(x),
        "bool": lambda x: "true" in x.lower(),
        "str": lambda x: x,
        "file": lambda x: x,
        "dir": lambda x: x,
        "tuple": lambda x: tuple([s.strip() for s in x.split(",")]),
    }

    # Parameter post checkers can be combined by an underscore between them, in which case all of those checkers must
    # be true for the parameter to be valid. These checkers are applied AFTER the parameter is formatted into its
    # correct type. The post checks are not done if the parameter can be "maybe" and is None.
    _param_post_checks: dict[str, tuple[Callable[[Any], bool]], str] = {
        "positive": (lambda x: x > 0, "positive"),
        "negative": (lambda x: x < 0, "negative"),
        "not-positive": (lambda x: x <= 0, "<= 0"),
        "not-negative": (lambda x: x >= 0, ">= 0"),
        "lt1": (lambda x: x < 1, "< 1"),
        "lteq1": (lambda x: x <= 1, "<= 1"),
        # String checks.
        "str-not-empty": (lambda x: bool(x.strip()), "a non empty string"),
        "file": (lambda x: path.isfile(x), "an existing file"),
        "dir": (lambda x: path.isdir(x), "an existing directory"),
        "email-address": (lambda x: re.match(r"^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$", x), "a valid email address"),
        # Tuple checks.
        "tup-not-empty": (lambda x: len(x) > 0, "a non empty tuple"),
        "len-multiple-3": (lambda x: len(x) % 3 == 0, "have length as a multiple of 3"),
    }
    _format_separator = "_"
    _checker_separator = "_"

    # Each key is a configuration section.
    # Each value is a dict with each key being a parameter name, the value is a tuple containing two strings. The first
    # specifies the format for pre-checks and formatting, the second specifies the post checks, like "positive". This
    # can be left empty.

    # If you change config options, update the config.default.ini file too.
    _options = {
        "basic_info": {
            "use_tiles": ("maybe_tuple_int", "tup-not-empty"),
            "use_rounds": ("maybe_tuple_int", "tup-not-empty"),
            "use_channels": ("maybe_tuple_int", "tup-not-empty"),
            "use_z": ("maybe_tuple_int", "tup-not-empty"),
            "use_dyes": ("maybe_tuple_int", "tup-not-empty"),
            "use_anchor": ("bool", ""),
            "anchor_round": ("maybe_int", "not-negative"),
            "anchor_channel": ("maybe_int", "not-negative"),
            "dapi_channel": ("maybe_int", "not-negative"),
            "dye_names": ("tuple_str", "tup-not-empty"),
            "is_3d": ("bool", ""),
            "bad_trc": ("maybe_tuple_int", "len-multiple-3"),
            "ignore_first_z_plane": ("bool", ""),
            "reverse_tile_positions_x": ("bool", ""),
            "reverse_tile_positions_y": ("bool", ""),
            # From here onwards these are not compulsory to enter and will be taken from the metadata
            # Only leaving them here to have backwards compatibility as Max thinks the user should influence these
            "channel_camera": ("maybe_tuple_int", ""),
            "channel_laser": ("maybe_tuple_int", ""),
        },
        "file_names": {
            "notebook_name": ("str", "str-not-empty"),
            "input_dir": ("str", "str-not-empty"),  # The raw data does not need to accessible once extract is complete.
            "output_dir": ("str", "dir_str-not-empty"),
            "tile_dir": ("str", "dir_str-not-empty"),
            "round": ("maybe_tuple_str", "tup-not-empty"),
            "anchor": ("maybe_str", "str-not-empty"),
            "raw_extension": ("str", "str-not-empty"),
            "raw_metadata": ("maybe_str", "str-not-empty"),
            "dye_camera_laser": ("maybe_file", ""),
            "code_book": ("str", "file"),
            "psf": ("maybe_str", ""),
            "pciseq": ("tuple_str", ""),
            "fluorescent_bead_path": ("maybe_str", ""),
            "initial_bleed_matrix": ("maybe_str", "file"),
        },
        "notifications": {
            "log_name": ("str", "str-not-empty"),
            "minimum_print_severity": ("int", "not-negative"),
            "sender_email": ("maybe_str", "str-not-empty_email-address"),
            "sender_email_password": ("maybe_str", "str-not-empty"),
            "email_me": ("maybe_str", "str-not-empty_email-address"),
        },
        "extract": {
            "num_rotations": ("int", "not-negative"),
            "z_plane_mean_warning": ("number", "not-negative"),
        },
        "filter": {
            "force_cpu": ("bool", ""),
            "r_dapi": ("maybe_int", "positive"),
            "deconvolve": ("bool", ""),
            "wiener_constant": ("number", "not-negative"),
            "wiener_pad_shape": ("tuple_int", "tup-not-empty"),
        },
        "find_spots": {
            "auto_thresh_multiplier": ("number", "not-negative"),
            "radius_xy": ("int", "not-negative"),
            "radius_z": ("int", "not-negative"),
            "max_spots_percent": ("number", "not-negative"),
            "n_spots_warn_fraction": ("number", "not-negative"),
            "n_spots_error_fraction": ("number", "not-negative"),
        },
        "stitch": {
            "expected_overlap": ("number", "not-negative"),
        },
        "register": {
            # this parameter is for channel registration
            "bead_radii": ("maybe_tuple_number", ""),
            # these parameters are for round registration
            "sample_factor_yx": ("int", "positive"),
            "chunks_yx": ("int", "positive"),
            "overlap_yx": ("number", "not-negative"),
            "window_radius": ("int", "positive"),
            "smooth_sigma": ("tuple_number", "tup-not-empty"),
            "flow_cores": ("maybe_int", "positive"),
            "flow_clip": ("maybe_tuple_number", "tup-not-empty"),
            # these parameters are for icp
            "neighb_dist_thresh_yx": ("number", "positive"),
            "neighb_dist_thresh_z": ("maybe_number", "positive"),
            "icp_min_spots": ("int", "not-negative"),
            "icp_max_iter": ("int", "not-negative"),
        },
        "call_spots": {
            "gene_prob_threshold": ("number", "not-negative"),
            "target_values": ("maybe_tuple_number", "tup-not-empty"),
            "d_max": ("maybe_tuple_int", "tup-not-empty"),
            "kappa": ("maybe_number", "not-negative"),
            "concentration_parameter_parallel": ("number", ""),
            "concentration_parameter_perpendicular": ("number", ""),
        },
        "omp": {
            "weight_coef_fit": ("bool", ""),
            "max_genes": ("int", "positive"),
            "minimum_intensity": ("number", "not-negative"),
            "dot_product_threshold": ("number", "not-negative"),
            "subset_pixels": ("maybe_int", "positive"),
            "force_cpu": ("bool", ""),
            "radius_xy": ("int", "positive"),
            "radius_z": ("int", "positive"),
            "mean_spot_filepath": ("maybe_str", "file"),
            "score_threshold": ("number", "not-negative"),
        },
        "thresholds": {
            "intensity": ("maybe_number", ""),
            "score_ref": ("number", ""),
            "score_omp": ("number", ""),
            "score_prob": ("number", ""),
            "score_omp_multiplier": ("number", ""),
        },
        "reg_to_anchor_info": {
            "full_anchor_y0": ("maybe_number", ""),
            "full_anchor_x0": ("maybe_number", ""),
            "partial_anchor_y0": ("maybe_number", ""),
            "partial_anchor_x0": ("maybe_number", ""),
            "side_length": ("maybe_number", ""),
        },
    }

    def __init__(self) -> None:
        pass

    def __getitem__(self, name: str) -> None:
        if type(name) is not str:
            raise TypeError(f"Config sections must be accessed through a string, got type {type(name)}")

        for section in self._sections:
            if section.name == name:
                return section

        raise ValueError(f"No config section named {name}")

    def load(self, file_path: str) -> None:
        """
        Load the configuration file from the given file path. Any unset configuration values are set to their default
        value specified in file coppafish/setup/settings.default.ini.

        Args:
            file_path (str): the file path to the custom configuration file set by the user.
        """
        assert type(file_path) is str
        if not path.isfile(file_path):
            raise FileNotFoundError(f"Could not find config file at {file_path}")

        parser = configparser.ConfigParser()
        # Make parameter names case-sensitive
        parser.optionxform = str

        # Load the default settings first.
        default_path = self.get_default_config_file_path()
        with open(default_path, "r") as f:
            parser.read_string(f.read())

        # Overwrite the default settings if the user has specified them.
        with open(file_path, "r") as f:
            parser.read_string(f.read())

        # Ensure every section in _options exists in the config files.
        for section in self._options.keys():
            if section not in parser.keys():
                raise self.SectionError(f"No config section {section} found in config _options")

        # Ensure there are no additional sections in the config files.
        for section in parser.keys():
            if section == "DEFAULT":
                continue
            if section not in self._options.keys():
                raise self.SectionError(f"Unexpected config section {section} that is not in _options.")

        param_msg = f"parameter {param_name} in section {section}"
        # Ensure every expected config parameter exists in the config files.
        for section in self._options.keys():
            for param_name in self._options[section].keys():
                if param_name not in parser[section].keys():
                    raise self.MissingParamError(f"Expected {param_msg}")

        # Run pre-checks on the parameters.
        for section in self._options.keys():
            for param_name, param_value in parser[section].items():
                if param_name not in self._options[section].keys():
                    log.warn(f"Unexpected config {param_msg}")
                    parser[section].__delitem__(param_name)
                    continue
                pre_checker_str = self._options[section][param_name][0]
                if not self.pre_check_param(param_name, section, param_value, pre_checker_str):
                    raise self.ParamError(
                        f"Failed check on {param_msg}, expected type "
                        + f"{self.convert_pre_check_to_readable(pre_checker_str)} but got {param_value}"
                    )

        # TODO: Run parameter formatting.

    def pre_check_param(self, name: str, section: str, value: Any, checker_str: str) -> bool:
        """
        Check if the given parameter value is valid based on its post-check string.

        Args:
            name (str): the name of the parameter.
            section (str): the name of the section the parameter is in.
            value (any): the value of the parameter, after formatting.
            checker_str (str): the keyword checks for the parameter, separated by underscores. For example, "positive",
                "positive_lt1" to be positive and less than 1.

        Returns:
            (bool): valid. Whether the parameter is valid or not.
        """
        assert type(checker_str) is str
        assert checker_str

        if not checker_str:
            return True

        for check_name in checker_str.split(self._checker_separator):
            if check_name not in self._param_post_checks.keys():
                raise ValueError(f"Unknown check {check_name} given to parameter {name} in {section} in _options")

            if not self._param_post_checks[check_name][0](value):
                return False
        return True

    def convert_pre_check_to_readable(self, pre_check_str: str) -> str:
        """
        Convert a pre-check string into a more readable format for debugging and user convenience.

        Args:
            pre_check_str (str): the pre check string.

        Returns:
            (str): pre_check_readable. An easier to read version of the pre-check.
        """
        if not pre_check_str:
            return "nothing"
        pre_check_readable = ""
        for sub_check in pre_check_str.split(self._checker_separator):
            if not sub_check:
                continue
            if sub_check == "maybe":
                continue
            if sub_check == "tuple":
                pre_check_readable += "a series of "
            else:
                pre_check_readable += sub_check
        if pre_check_str.startswith("maybe_"):
            pre_check_readable += " or empty"

        return pre_check_readable

    def format_param(self, name: str, section: str, value: str, format_str: str) -> Any:
        """
        Format the given parameter value based on the given formatting string set for said parameter. For example, if
        the parameter is "0.1" the format_str is "number" then it is converted to a float.

        Args:
            name (str): the parameter's name.
            section (str): the config section the parameter is located in.
            value (str): the string representation of the parameter.
            format_str (str): the specified formatting for the parameter. Some examples are "maybe_int", "int",
                "number", "maybe_tuple_int", "tuple_str". Tuples cannot be nested like notebook page variables.

        Returns:
            (any): formatted_value. The value formatted into the correct type.
        """
        assert type(name) is str
        assert type(section) is str
        assert type(value) is str
        assert type(format_str) is str

        # Deal with weirder formatting first, if they exist.
        if "maybe" in format_str and value == "":
            assert format_str.startswith("maybe") and format_str.count("maybe") == 1
            return None
        formatted_value: tuple[str, ...] = (value,)
        keep_tuple = False
        if "tuple" in format_str:
            assert format_str.count("tuple") == 1
            formatted_value: tuple[str, ...] = self._param_formatters["tuple"](value)
            keep_tuple = True

        format_str = format_str.replace("maybe_", "").replace("tuple_", "")
        for sub_format in format_str.split(self._format_separator):
            if not sub_format:
                continue
            if sub_format not in self._param_formatters.keys():
                raise ValueError(f"Unknown format {sub_format} specified for config parameter {name} in {section}")
            new_formatted_value = tuple()
            for sub_value in formatted_value:
                try:
                    new_formatted_value += (self._param_formatters[sub_format](sub_value),)
                except ValueError as e:
                    raise self.ParamError(
                        f"Failed to format config parameter {name} in {section} with formatting {sub_format}. It has "
                        + f"value {sub_value}. Error given: {e}"
                    )
            formatted_value: tuple[Any, ...] = new_formatted_value

        if not keep_tuple:
            assert len(formatted_value) == 1
            formatted_value = formatted_value[0]
        return formatted_value

    def post_check_param(self, name: str, section: str, value: Any, checker_str: str) -> bool:
        """
        Check if the given parameter value is valid based on its post-check string.

        Args:
            name (str): the name of the parameter.
            section (str): the name of the section the parameter is in.
            value (any): the value of the parameter, after formatting.
            checker_str (str): the keyword checks for the parameter, separated by underscores. For example, "positive",
                "positive_lt1" to be positive and less than 1.

        Returns:
            (bool): valid. Whether the parameter is valid or not.
        """
        assert type(checker_str) is str
        assert checker_str

        if not checker_str:
            return True

        for check_name in checker_str.split(self._checker_separator):
            if check_name not in self._param_post_checks.keys():
                raise ValueError(f"Unknown check {check_name} given to parameter {name} in {section} in _options")

            if not self._param_post_checks[check_name][0](value):
                return False
        return True

    def _get_default_file_path(self) -> str:
        return str(importlib_resources.files("coppafish.setup").joinpath("default.ini"))

    class ParamError(Exception):
        pass

    class MissingParamError(Exception):
        pass

    class SectionError(Exception):
        pass
