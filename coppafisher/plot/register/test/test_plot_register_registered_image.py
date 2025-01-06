from os import path

import pytest

from coppafisher import Notebook
from coppafisher.plot import view_registered_images


@pytest.mark.notebook
def test_view_registered_images() -> None:
    nb_path = path.dirname(path.dirname(path.dirname(path.dirname(__file__))))
    nb_path = path.join(nb_path, "robominnie", "test", ".integration_dir", "output_coppafisher", "notebook")
    nb = Notebook(nb_path)
    view_registered_images(nb, show=False)
