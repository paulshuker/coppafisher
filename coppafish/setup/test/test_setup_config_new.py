import os
import tempfile

import numpy as np

from coppafish.setup import config_new


def test_Config() -> None:
    config = config_new.Config()
    config._options.clear()
    config._options["debug"] = {
        "1": ("int", ""),
        "2": ("number", ""),
        "3": ("number", ""),
        "4": ("bool", ""),
        "5": ("file", ""),
        "6": ("dir", ""),
        "7": ("tuple_int", ""),
        "8": ("tuple_number", ""),
        "9": ("tuple_number", ""),
        "10": ("tuple_bool", ""),
        "11": ("tuple_file", ""),
        "12": ("tuple_dir", ""),
        "13": ("maybe_int", ""),
        "14": ("maybe_number", ""),
        "15": ("maybe_str", ""),
        "16": ("maybe_bool", ""),
        "17": ("maybe_file", ""),
        "18": ("maybe_dir", ""),
        "19": ("maybe_tuple_int", ""),
        "20": ("maybe_tuple_number", ""),
        "21": ("maybe_tuple_number", ""),
        "22": ("maybe_tuple_bool", ""),
        "23": ("maybe_tuple_file", ""),
        "24": ("maybe_tuple_dir", ""),
    }
    tmpdir = tempfile.TemporaryDirectory()
    tmpdir2 = tempfile.TemporaryDirectory()
    tmpfile = tempfile.TemporaryFile()
    tmpfile2 = tempfile.TemporaryFile()
    config_filepath = os.path.join(tmpdir.name, f"test_config.ini")

    # Build a default config file with all parameters wrongly assigned.
    default_config_filepath = os.path.join(tmpdir.name, "default.ini")
    default_config_content = "[debug]\n"
    for i in range(1, 25):
        default_config_content += f"{i}  =   "
        default_config_content += "kfhwjghe4534, 242ref\n"

    rng = np.random.RandomState(0)

    with open(default_config_filepath, "w") as file:
        file.write(default_config_content)

    config_content = "\n[debug]\n"
    for i in range(1, 25):
        expected_type = config._options["debug"][str(i)][0]
        if "maybe" in expected_type and rng.rand() > 0.9:
            config_content += f"{i} =    \n"
        elif "tuple_int" in expected_type:
            config_content += f"{i} = 0, 2, -4 \n"
        elif "tuple_number" in expected_type:
            config_content += f"{i} = 0, 2.203, -4.5   \n"
        elif "tuple_str" in expected_type:
            config_content += f"{i} = acb, gaming, bruh \n"
        elif "tuple_bool" in expected_type:
            config_content += f"{i} = true, TrUE, FaLsE, false, FALSE \n"
        elif "tuple_file" in expected_type:
            config_content += f"{i} =  {tmpfile.name}, {tmpfile2.name}, {tmpfile.name} \n"
        elif "tuple_dir" in expected_type:
            config_content += f"{i} = {tmpdir2.name}, {tmpdir.name}  \n"
        elif "int" in expected_type:
            config_content += f"{i} = {'+' if rng.rand()>0.5 else '-'}35533  \n"
        elif "number" in expected_type:
            config_content += f"{i} = {'+' if rng.rand()>0.5 else '-'}35533.32552  \n"
        elif "str" in expected_type:
            config_content += f"{i} =  edsffegeg sffefe  \n"
        elif "bool" in expected_type:
            config_content += f"{i} = TrUe\n"
        elif "file" in expected_type:
            config_content += f"{i} = {tmpfile.name}\n"
        elif "dir" in expected_type:
            config_content += f"{i} = {tmpdir2.name}\n"
        else:
            assert False, "I forgot something"

    # Set only the ith parameter incorrectly and assert an error is raised when loading the config.
    for i in range(1, 25):
        wrong_param_name = str(i)
        ind_min = config_content.index(wrong_param_name + " =")
        ind_max = config_content.index("\n", ind_min)
        config_content_wrong = config_content[:ind_min]
        config_content_wrong += wrong_param_name + "   ="
        config_content_wrong += "sfefeg353sfe".replace(str(i), "")
        config_content_wrong += config_content[ind_max:]
        with open(config_filepath, "w") as config_file:
            config_file.write(config_content_wrong)
        # Expect as error here as the default config does not have the debug section.
        try:
            config.load(config_filepath)
            assert False, "Expect SectionError"
        except config.SectionError:
            pass

        try:
            config.load(config_filepath, "/wrong/path")
            assert False, "Expect FileNotFoundError"
        except FileNotFoundError:
            pass

        try:
            config.load(config_filepath, default_config_filepath)
            assert False, "Expect ParamError"
        except config.ParamError:
            pass

    # TODO: Test str, tuple_str, and maybe_tuple_str separately.

    tmpdir.cleanup()
    tmpdir2.cleanup()
    tmpfile.close()
    tmpfile2.close()


if __name__ == "__main__":
    test_Config()
