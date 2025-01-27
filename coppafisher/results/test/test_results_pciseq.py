from os import path

import pandas as pd
import pytest

from coppafisher.omp import base
from coppafisher.results import pciseq
from coppafisher.setup.notebook import Notebook


@pytest.mark.notebook
def test_export_to_pciseq() -> None:
    nb_path = path.dirname(path.dirname(path.dirname(__file__)))
    nb_path = path.join(nb_path, "robominnie", "test", ".integration_dir", "output_coppafisher", "notebook")

    nb = Notebook(nb_path)

    # Check that export to pciseq does not crash and that a csv file appears for each gene call method.
    for method in ("prob", "anchor", "omp"):
        csv_file_path = pciseq.export_to_pciseq(nb, method)
        assert type(csv_file_path) is str
        assert path.isfile(csv_file_path)
        assert method in csv_file_path
        assert csv_file_path.endswith(".csv")

        n_expected_genes = (
            base.get_all_gene_no(nb.basic_info, nb.omp)[0].size if method == "omp" else nb.ref_spots.tile.size
        )
        results = pd.read_csv(csv_file_path)
        assert len(results) == n_expected_genes

        csv_file_path = pciseq.export_to_pciseq(nb, method, 0.5)
        assert type(csv_file_path) is str
        assert path.isfile(csv_file_path)
        assert method in csv_file_path
        assert csv_file_path.endswith(f"pciseq_{method}.csv")

        if method == "prob":
            n_expected_genes = (nb.call_spots.gene_probabilities[:].max(1) >= 0.5).sum()
        elif method == "anchor":
            n_expected_genes = (nb.call_spots.dot_product_gene_score[:] >= 0.5).sum()
        elif method == "omp":
            n_expected_genes = (base.get_all_scores(nb.basic_info, nb.omp)[0] >= 0.5).sum()
        results = pd.read_csv(csv_file_path)
        assert len(results) == n_expected_genes
        assert csv_file_path.endswith(f"pciseq_{method}.csv")

        csv_file_path = pciseq.export_to_pciseq(nb, method, 0.5, 0.7)
        assert type(csv_file_path) is str
        assert path.isfile(csv_file_path)
        assert method in csv_file_path
        assert csv_file_path.endswith(f"pciseq_{method}.csv")
