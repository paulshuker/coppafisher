import numpy as np


class Gene:
    """
    Data for a single gene.

    The data stored is used by the viewer and the gene legend to keep track of visible and hidden genes.
    """

    name: str
    notebook_index: int
    colour: np.ndarray
    symbol_napari: str
    cell_type: str
    active: bool

    def __init__(
        self,
        name: str,
        notebook_index: int,
        colour: np.ndarray,
        symbol_napari: str,
        cell_type: str,
        active: bool = True,
    ):
        """
        Create gene data.

        Args:
            name (str): gene name.
            notebook_index (int): index of the gene within the notebook.
            colour (`(3) ndarray`): with the RGB colour of the gene.
            symbol_napari (str): symbol used to plot in napari.
            cell_type (str): name of the cell class most associated with the gene.
            active (bool, optional): whether the gene is currently visible in the Viewer. Used for toggling gene
                visibility. Default: true.
        """
        self.name = name
        self.notebook_index = notebook_index
        self.colour = colour
        self.symbol_napari = symbol_napari
        self.cell_type = cell_type
        self.active = active
