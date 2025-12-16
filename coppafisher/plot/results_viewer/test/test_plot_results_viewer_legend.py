import matplotlib
import numpy as np

from coppafisher.plot.results_viewer import gene, legend
from coppafisher.plot.results_viewer.subplot import Subplot


def test_Legend() -> None:
    matplotlib.use("Agg")

    rng = np.random.RandomState(0)

    l = legend.Legend()
    assert isinstance(l, Subplot)

    n_genes = 17
    genes = []
    for i in range(n_genes):
        genes.append(
            gene.Gene(
                f"{i}",
                i,
                rng.rand(3),
                list(l._napari_to_mpl_marker.keys())[rng.randint(len(l._napari_to_mpl_marker))],
                f"g{i}",
            )
        )
    genes = tuple(genes)

    l.create_gene_legend(genes, "cell_type")
    active_genes: list[bool] = [True if rng.rand() > 0.5 else False for _ in range(n_genes)]
    l.update_selected_legend_genes(active_genes)

    assert type(l.get_help()) is tuple
    assert all(type(item) is str for item in l.get_help())
