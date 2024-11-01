import torch

from coppafish.utils import duplicates


def test_is_duplicate_spot() -> None:
    yxz_global_positions = torch.zeros((5, 3)).int()
    yxz_global_positions[0, 0] = 8
    yxz_global_positions[0, 1] = 2
    yxz_global_positions[0, 2] = 2
    yxz_global_positions[2, 0] = -1
    yxz_global_positions[2, 1] = 20
    yxz_global_positions[2, 2] = 2
    yxz_global_positions[3, 0] = -1
    yxz_global_positions[3, 1] = 11
    yxz_global_positions[3, 2] = 0
    yxz_global_positions[4, 0] = 7
    yxz_global_positions[4, 1] = 0
    yxz_global_positions[4, 2] = 1
    tile_number = 0
    tile_centres = torch.zeros((4, 3)).float()
    tile_centres[0, 0] = 7
    tile_centres[0, 1] = 0
    tile_centres[0, 2] = 1
    tile_centres[2, 0] = 0
    tile_centres[2, 1] = 10
    tile_centres[3, 0] = 99_999
    tile_centres[3, 1] = 99_999
    tile_centres[3, 2] = 99_999
    is_duplicate = duplicates.is_duplicate_spot(yxz_global_positions, tile_number, tile_centres)
    assert type(is_duplicate) is torch.Tensor
    assert is_duplicate.shape == (yxz_global_positions.shape[0],)
    assert not is_duplicate[0]
    assert is_duplicate[1]
    assert is_duplicate[2]
    assert is_duplicate[3]
    assert not is_duplicate[4]
    assert type(yxz_global_positions) is torch.Tensor
    assert type(tile_centres) is torch.Tensor
