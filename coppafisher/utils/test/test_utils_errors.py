import numpy as np

from coppafisher.utils import errors


def test_compare_spots() -> None:
    spot_positions_0 = np.zeros((0, 3), np.float32)
    spot_gene_indices_0 = np.zeros(0, np.int16)
    spot_positions_1 = np.zeros((0, 3), np.float32)
    spot_gene_indices_1 = np.zeros(0, np.int16)
    distance_threshold = 0.1
    assignments, FNs = errors.compare_spots(
        spot_positions_0, spot_gene_indices_0, spot_positions_1, spot_gene_indices_1, distance_threshold
    )
    assert ((assignments >= 0) & (assignments <= 2)).all()
    TPs = (assignments == 0).sum()
    WPs = (assignments == 1).sum()
    FPs = (assignments == 2).sum()
    assert TPs == WPs == FPs == FNs == 0
    spot_positions_0 = np.zeros((2, 3), np.float32)
    spot_positions_0[0] = [3.75, 0, 0]
    spot_gene_indices_0 = np.zeros(2, np.int16)
    spot_positions_1 = np.zeros((1, 3), np.float32)
    spot_gene_indices_1 = np.zeros(1, np.int16)
    distance_threshold = 3.7
    assignments, FNs = errors.compare_spots(
        spot_positions_0, spot_gene_indices_0, spot_positions_1, spot_gene_indices_1, distance_threshold
    )
    assert ((assignments >= 0) & (assignments <= 2)).all()
    TPs = (assignments == 0).sum()
    WPs = (assignments == 1).sum()
    FPs = (assignments == 2).sum()
    assert TPs == 1
    assert WPs == 0
    assert FPs == 1
    assert FNs == 0
    spot_gene_indices_0[1] = 1
    assignments, FNs = errors.compare_spots(
        spot_positions_0, spot_gene_indices_0, spot_positions_1, spot_gene_indices_1, distance_threshold
    )
    assert ((assignments >= 0) & (assignments <= 2)).all()
    TPs = (assignments == 0).sum()
    WPs = (assignments == 1).sum()
    FPs = (assignments == 2).sum()
    assert TPs == 0
    assert WPs == 1
    assert FPs == 1
    assert FNs == 0
    # False negative spot example.
    spot_positions_0 = np.zeros((1, 3), np.float32)
    spot_gene_indices_0 = np.zeros(1, np.int16)
    spot_positions_1 = np.zeros((2, 3), np.float32)
    spot_positions_1[1] = [0.1, 0.5, 10]
    spot_gene_indices_1 = np.zeros(2, np.int16)
    distance_threshold = 7.1
    assignments, FNs = errors.compare_spots(
        spot_positions_0, spot_gene_indices_0, spot_positions_1, spot_gene_indices_1, distance_threshold
    )
    assert ((assignments >= 0) & (assignments <= 2)).all()
    TPs = (assignments == 0).sum()
    WPs = (assignments == 1).sum()
    FPs = (assignments == 2).sum()
    assert TPs == 1
    assert WPs == 0
    assert FPs == 0
    assert FNs == 1
    # Too many matching spots example.
    spot_positions_0 = np.zeros((5, 3), np.float32)
    spot_gene_indices_0 = np.zeros(5, np.int16)
    spot_positions_1 = np.zeros((2, 3), np.float32)
    spot_gene_indices_1 = np.zeros(2, np.int16)
    distance_threshold = 1.0
    assignments, FNs = errors.compare_spots(
        spot_positions_0, spot_gene_indices_0, spot_positions_1, spot_gene_indices_1, distance_threshold
    )
    assert ((assignments >= 0) & (assignments <= 2)).all()
    TPs = (assignments == 0).sum()
    WPs = (assignments == 1).sum()
    FPs = (assignments == 2).sum()
    assert TPs == 2
    assert WPs == 0
    assert FPs == 3
    assert FNs == 0
