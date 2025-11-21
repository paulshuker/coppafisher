import os
import warnings

import pytest

from coppafisher import Notebook, RegistrationViewer, Viewer
from coppafisher.robominnie.robominnie import Robominnie


def get_robominnie_scores(rm: Robominnie) -> None:
    tile_scores = rm.score_tiles("prob", score_threshold=0.9, intensity_threshold=0.4)
    print(f"Prob scores for each tile: {tile_scores}")
    if any([score < 75 for score in tile_scores]):
        warnings.warn("Anchor method contains tile score < 75%", stacklevel=1)
    if any([score < 40 for score in tile_scores]):
        raise ValueError("Anchor method has a tile score < 40%. This can be a sign of a pipeline bug")

    tile_scores = rm.score_tiles("anchor", score_threshold=0.5, intensity_threshold=0.4)
    print(f"Anchor scores for each tile: {tile_scores}")
    if any([score < 75 for score in tile_scores]):
        warnings.warn("Anchor method contains tile score < 75%", stacklevel=1)
    if any([score < 40 for score in tile_scores]):
        raise ValueError("Anchor method has a tile score < 40%. This can be a sign of a pipeline bug")

    tile_scores = rm.score_tiles("omp", score_threshold=0.90, intensity_threshold=0.813)
    print(f"OMP scores for each tile: {tile_scores}")
    if any([score < 75 for score in tile_scores]):
        warnings.warn("OMP method contains tile score < 75%", stacklevel=1)
    if any([score < 40 for score in tile_scores]):
        raise ValueError("OMP method has a tile score < 40%. This can be a sign of a pipeline bug")


@pytest.mark.slow
def test_integration_2d_two_tiles():
    """
    Summary of input data: random spots and pink noise.

    Includes anchor round, sequencing rounds, two `1x100x100` tiles.
    """
    output_dir = get_output_dir()
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    robominnie = Robominnie(n_channels=5, n_planes=1, tile_sz=128, n_tiles_y=2)
    robominnie.generate_gene_codes()
    robominnie.generate_pink_noise()
    robominnie.add_spots()
    robominnie.save_raw_images(output_dir)
    robominnie.run_coppafisher()
    robominnie.score_tiles("prob", 0.9, 0.4)
    robominnie.score_tiles("anchor", 0.5, 0.4)
    robominnie.score_tiles("omp", 0.05, 0.4)
    robominnie.view_spot_positions()
    get_robominnie_scores(robominnie)
    del robominnie


@pytest.mark.integration
def test_integration_small_two_tiles():
    """
    Summary of input data: random spots and pink noise.

    Includes anchor round, sequencing rounds, two `10x128x128` tiles.
    """
    output_dir = get_output_dir()
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    robominnie = Robominnie(n_channels=5, n_planes=10, tile_sz=128, n_tiles_y=2)
    robominnie.generate_gene_codes()
    robominnie.generate_pink_noise()
    robominnie.add_spots()
    # We add a fake bad tile, round, channel image to make sure it can run through the pipeline.
    robominnie.save_raw_images(output_dir, bad_trcs=[[0, 2, 3]])
    robominnie.run_coppafisher()
    get_robominnie_scores(robominnie)
    del robominnie


def viewers_test() -> None:
    """
    Make sure the coppafisher Viewer and RegistrationViewer is working without crashing.

    Notes:
        - Requires a robominnie instance to have successfully run through first.
    """
    notebook_path = get_notebook_path()
    if not os.path.exists(notebook_path):
        return
    gene_colours_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".integration_dir/gene_colours.csv")
    notebook = Notebook(notebook_path)
    Viewer(notebook, gene_marker_filepath=gene_colours_path)
    RegistrationViewer(notebook, get_config_path())


def get_output_dir() -> str:
    return os.path.dirname(os.path.dirname(get_notebook_path()))


def get_notebook_path() -> str:
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), ".integration_dir/output_coppafisher/notebook")


def get_config_path() -> str:
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), ".integration_dir/robominnie.ini")


if __name__ == "__main__":
    # test_integration_small_two_tiles()
    test_integration_2d_two_tiles()
    viewers_test()
