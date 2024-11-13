from collections.abc import Iterable
from typing import Any, Tuple


class ConfigSection:
    """
    Stores a config section's variables.

    The config section has no for formatting and type checking config values. This is the job of the config itself. The
    config section just accepts whatever it is given.

    A config section will keep track of how many times a config parameter is gathered. So, if a config parameter is
    never used, this can be detected by the function `list_redundant_params`.

    Attributes:
        name (str): the name of the config section.
    """

    name: str

    _attributes: dict[str, Any]
    _retrieval_counts: dict[str, int]

    def __init__(self, name: str, values: dict[str, Any]) -> None:
        """
        Initialise a config section.

        Args:
            name (str): the name of the config section.
            values (dict[str, any]): the name of each parameter as a key, the value as its value in the config section.
        """
        assert type(name) is str
        assert type(values) is dict
        assert all([type(param_name) is str for param_name in values.keys()])
        assert len(values) > 0, "Config must contain at least one parameter"

        self.name = name
        self._attributes = {}
        self._retrieval_counts = {name: 0 for name in values.keys()}
        for name, val in values.items():
            self._attributes[name] = val

    def __getitem__(self, name: str) -> Any:
        if type(name) is not str:
            raise TypeError(f"Config section parameters must be accessed individually by a str, got {type(name)}")
        if name not in self._attributes:
            raise ValueError(f"Could not find parameter {name} in {self.name} config section")

        self._retrieval_counts[name] += 1
        return self._attributes[name]

    def items(self) -> Iterable[Tuple[str, Any]]:
        """
        List all the config section parameter names and values.

        Returns:
            (iterable[tuple[str, any]]): items. All config section parameter names and values.
        """
        return self._attributes.items()

    def list_redundant_params(self) -> list[str]:
        """
        List parameters that were never retrieved in the config section.

        Returns:
            (list[int]): redundant_names. A list containing the names of each parameter never retrieved.
        """
        redundant_parameters = []
        for name, count in self._retrieval_counts.items():
            if count == 0:
                redundant_parameters.append(name)
        return redundant_parameters
