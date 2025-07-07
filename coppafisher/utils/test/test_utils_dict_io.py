import os
import pickle
import tempfile

from coppafisher.utils import dict_io


def test_save_dict_and_try_load_dict() -> None:
    dictionary = {"a": 2, "chs": "b", "c": True}

    tmp_dir = tempfile.TemporaryDirectory("coppafisher")

    wrong_file_path = os.path.join(tmp_dir.name, "dict_test_2.pkl")
    file_path = os.path.join(tmp_dir.name, "dict_test.pkl")
    assert not os.path.isfile(file_path)
    dict_io.save_dict(dictionary, file_path)
    assert os.path.isfile(file_path)

    with open(file_path, "rb") as file:
        dictionary_output = pickle.load(file)

    assert type(dictionary_output) is dict
    assert len(dictionary_output) == 3
    assert dictionary_output["a"] == 2
    assert dictionary_output["chs"] == "b"
    assert dictionary_output["c"]
    assert dictionary_output == dictionary

    dictionary_output = dict_io.try_load_dict(wrong_file_path, {"b": 4})
    assert len(dictionary_output) == 1
    assert dictionary_output["b"] == 4

    dictionary_output = dict_io.try_load_dict(file_path, {"b": 4})
    assert type(dictionary_output) is dict
    assert len(dictionary_output) == 3
    assert dictionary_output["a"] == 2
    assert dictionary_output["chs"] == "b"
    assert dictionary_output["c"]
    assert dictionary_output == dictionary

    tmp_dir.cleanup()
