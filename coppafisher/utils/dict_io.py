import os
import pickle


def save_dict(dictionary: dict, file_path: str) -> None:
    """
    Save the dictionary to the given file path.

    The dictionary is saved as a pickle (.pkl) file type.

    Args:
        dictionary (dict): the dictionary to save.
        file_path (str): the save file path, must end in ".pkl".
    """
    assert type(dictionary) is dict
    assert type(file_path) is str
    assert file_path.endswith(".pkl")

    with open(file_path, "wb") as file:
        pickle.dump(dictionary, file)


def try_load_dict(file_path: str, default: dict) -> dict:
    """
    Try to load a dictionary at file_path.

    The dictionary must be saved as a pickle (.pkl) file type. If the file does not exist, then default is returned
    instead.

    Args:
        file_path (str): the file path, must end in ".pkl".
        default (dict): what is returned if the file does not exist.

    Raises:
        TypeError: if the loaded file at file_path is not a dict type.
    """
    assert type(file_path) is str
    assert file_path.endswith(".pkl")
    assert type(default) is dict

    if not os.path.isfile(file_path):
        return default

    output = None
    with open(file_path, "rb") as file:
        output = pickle.load(file)

    if type(output) is not dict:
        raise TypeError(f"File at {file_path} is not a dict type, got {type(output)} instead")

    return output
