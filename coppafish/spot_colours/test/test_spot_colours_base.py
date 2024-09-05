import numpy as np
import scipy
import skimage
import torch

from coppafish.spot_colours import base as spot_colours_base
from coppafish.spot_colours.base import get_spot_colours


def test_convert_coords_to_torch_grid() -> None:
    # Test this function by grid sampling an image for all its pixel positions, so no interpolation should be done and
    # the same image should be given back.
    rng = np.random.RandomState(0)
    image_shape = 4, 5, 6
    image = rng.rand(*image_shape)
    yxz_coords = np.meshgrid(
        np.linspace(0, image_shape[0] - 1, image_shape[0]),
        np.linspace(0, image_shape[1] - 1, image_shape[1]),
        np.linspace(0, image_shape[2] - 1, image_shape[2]),
        indexing="ij",
    )
    # NOTE: Order 'F' reshaping must be used in numpy reshape method as this is the same as pytorch's reshape method.
    yxz_coords = np.array(yxz_coords).T.reshape((-1, 3), order="F")
    yxz_coords = torch.tensor(yxz_coords)
    yxz_grid_coords = spot_colours_base.convert_coords_to_torch_grid(yxz_coords, image_shape)
    assert type(yxz_grid_coords) is torch.Tensor
    assert yxz_grid_coords.shape == (np.prod(image_shape).item(), 3)
    yxz_grid_coords = yxz_grid_coords.reshape(image_shape + (3,))[None]
    image = torch.tensor(image).float()
    # Input has shape (1, 1, 4, 5, 6).
    # Grid has shape (1, 4, 5, 6, 3).
    # Result has shape (1, 1, 4, 5, 6).
    grid_image = torch.nn.functional.grid_sample(image[None, None], yxz_grid_coords, align_corners=True)
    grid_image = grid_image[0, 0]
    assert grid_image.shape == image_shape
    assert torch.allclose(grid_image, image)


def test_apply_flow_new() -> None:
    # Check that a simple +1, +2, and -3 shift in each direction works.
    yxz = np.zeros((2, 3), dtype=np.int16)
    yxz[1] = [3, 5, 6]
    n_tiles, n_rounds = 1, 3
    tile = 0
    round = 1
    image_shape = (4, 5, 6)
    flow = np.zeros((n_tiles, n_rounds, 3) + image_shape)
    flow[tile, round, 0] = +1
    flow[tile, round, 1] = +2
    flow[tile, round, 2] = -3
    yxz_expected = yxz.copy()
    yxz_expected[0] = 1, 2, -3
    yxz_expected[1] = 4, 7, 3
    yxz_expected = torch.tensor(yxz_expected).float()
    yxz_flow = spot_colours_base.apply_flow_new(yxz, flow, tile, round)
    assert type(yxz_flow) is torch.Tensor
    assert yxz_flow.shape == (2, 3)
    assert torch.allclose(yxz_flow, yxz_expected)
    yxz = torch.tensor(yxz)
    yxz_flow = spot_colours_base.apply_flow_new(yxz, flow, tile, round)
    assert type(yxz_flow) is torch.Tensor
    assert yxz_flow.shape == (2, 3)
    assert torch.allclose(yxz_flow, yxz_expected)


def test_get_spot_colours_new() -> None:
    # Make sure that when the transforms are all identity the image is returned unchanged.
    rng = np.random.RandomState(0)
    n_tiles = 2
    n_rounds = 3
    n_channels = 4
    tile_shape = (5, 6, 7)
    image_shape = (n_tiles, n_rounds, n_channels) + tile_shape
    image = rng.rand(*image_shape).astype(np.float32)
    yxz = np.meshgrid(
        np.linspace(0, tile_shape[0] - 1, tile_shape[0]),
        np.linspace(0, tile_shape[1] - 1, tile_shape[1]),
        np.linspace(0, tile_shape[2] - 1, tile_shape[2]),
        indexing="ij",
    )
    yxz = np.array(yxz).T.reshape((-1, 3), order="F")
    flow = np.zeros((n_tiles, n_rounds, 3) + tile_shape)
    affine = np.zeros((n_tiles, n_rounds, n_channels, 4, 3))
    affine[:, :, :, :3] = np.eye(3)
    tile = 0
    output_dtype = np.float32
    colours = spot_colours_base.get_spot_colours_new(yxz, image, flow, affine, tile, output_dtype=output_dtype)
    assert type(colours) is np.ndarray
    assert colours.shape == (np.prod(tile_shape), n_rounds, n_channels)
    colours = colours.swapaxes(0, 1).swapaxes(1, 2).reshape((n_rounds, n_channels) + tile_shape, order="F")
    # TODO: Broken as usual
    assert np.allclose(colours, image[tile])


if __name__ == "__main__":
    test_convert_coords_to_torch_grid()
    test_apply_flow_new()
    test_get_spot_colours_new()


def test_get_spot_colours():
    """
    Function to test the get_spot_colours function from the spot_colours.base module.
    """
    # create some artificial data with 2 rounds, 3 channels and a 10 x 10 x 5 image
    rng = np.random.RandomState(0)
    tile_shape = 100, 100, 10
    n_rounds, n_channels = 2, 3
    images_aligned = rng.rand(n_rounds, n_channels, *tile_shape)
    # set values below 0.5 to 0, and above 0.5 to 1
    images_aligned = (images_aligned > 0.5).astype(np.float32)
    # smooth each round and channel independently
    for r in range(n_rounds):
        for c in range(n_channels):
            images_aligned[r, c] = skimage.filters.gaussian(images_aligned[r, c], sigma=[5, 5, 1])

    # now we would like to move these images by applying the inverse transform of the one we want to apply
    # to the spot colours and then check if we can recover the original images
    affine = np.zeros((n_channels, 4, 3))
    affine[:, :3, :3] = np.eye(3)
    # repeat the affine transforms for each round
    affine = np.repeat(affine[None], n_rounds, axis=0)
    # initialise flow to be 0
    flow = np.zeros((n_rounds, 3, *tile_shape))

    # 0. check that grabbing a single spot works
    spot_colours = get_spot_colours(
        image=images_aligned[None], flow=flow[None], affine_correction=affine[None], yxz_base=np.zeros((1, 3)), tile=0
    )
    assert spot_colours.shape == (1, n_rounds, n_channels)
    assert np.allclose(spot_colours, images_aligned[None, :, :, 0, 0, 0])

    # 1. check that the images are the same as the spot colours (no affine or flow applied)
    # get coords
    coords = np.array(np.meshgrid(*[np.arange(s) for s in tile_shape], indexing="ij"))
    yxz_base = coords.reshape(3, -1).T
    spot_colours = get_spot_colours(
        image=images_aligned[None], flow=flow[None], affine_correction=affine[None], yxz_base=yxz_base, tile=0
    )
    # reshape spot colours from n_spots x n_rounds x n_channels to n_y x n_x x n_z x n_rounds x n_channels
    spot_colours = spot_colours.reshape(*tile_shape, n_rounds, n_channels)
    # reorder spot colours array to n_rounds x n_channels x n_y x n_x x n_z
    spot_colours = np.transpose(spot_colours, (3, 4, 0, 1, 2))
    mid_z = tile_shape[2] // 2

    # # plot scatter plot of true vs predicted values for mid_z slice
    # import matplotlib.pyplot as plt
    # plt.scatter(x=images_aligned[:, :, :, :, mid_z].flatten(), y=spot_colours[:, :, :, :, mid_z].flatten())
    # plt.xlabel('True Values')
    # plt.ylabel('Predicted Values')
    # plt.title('True vs Predicted Values')
    # plt.show()
    #
    # # open napari viewer to check the images
    # import napari
    # v = napari.Viewer()
    # v.add_image(images_aligned, name='Aligned Images', colormap='red', blending='additive',
    #             contrast_limits=np.nanpercentile(images_aligned, [1, 99]))
    # v.add_image(spot_colours, name='Spot Colours', colormap='green', blending='additive',
    #             contrast_limits=np.nanpercentile(spot_colours, [1, 99]))
    # v.add_image(images_aligned - spot_colours, name='Difference',
    #             contrast_limits=np.nanpercentile(images_aligned - spot_colours, [1, 99]))
    # napari.run()

    # check that the spot colours are the same as the original images
    assert np.allclose(images_aligned, spot_colours, atol=1e-6)

    # 2. check that the images are the same as the spot colours (affine applied + flow applied)
    # set these affine transforms to be shifts in y and x by 1, 2, 3 and scales in y and x by 0.9, 0.8
    # these are the transforms we need to apply to go from anchor to target, so we need to apply the inverse to the
    # images
    affine[:, :, 3, 0] = [1, 2, 3]
    affine[:, :, 3, 1] = [1, 2, 3]
    affine[:, :, 0, 0] = 0.9
    affine[:, :, 1, 1] = 0.8

    # set flow to be +1 shift in z for round 0 and +2 shift in z for round 1
    flow[0, 2] = 1
    flow[1, 2] = 2

    # define the inverse warp
    warp_inv = np.array([coords - flow[r] for r in range(n_rounds)])

    # the transform we will apply to align is A(F(x)), so to disalign apply A^(-1)(F^(-1)(x))
    images_disaligned = np.zeros_like(images_aligned)
    for r in range(n_rounds):
        for c in range(n_channels):
            # invert the affine transform
            affine_rc = np.vstack([affine[r, c].T, [0, 0, 0, 1]])
            affine_rc = np.linalg.inv(affine_rc)
            images_disaligned[r, c] = scipy.ndimage.affine_transform(
                images_aligned[r, c], affine_rc, order=1, cval=np.nan
            )
            images_disaligned[r, c] = skimage.transform.warp(images_disaligned[r, c], warp_inv[r], order=1, cval=np.nan)

    # now we want to get the spot colours from the disaligned images
    spot_colours = get_spot_colours(
        image=images_disaligned[None], flow=flow[None], affine_correction=affine[None], yxz_base=yxz_base, tile=0
    )
    # reshape spot colours from n_spots x n_rounds x n_channels to n_y x n_x x n_z x n_rounds x n_channels
    spot_colours = spot_colours.reshape(*tile_shape, n_rounds, n_channels)
    # reorder spot colours array to n_rounds x n_channels x n_y x n_x x n_z
    spot_colours = np.transpose(spot_colours, (3, 4, 0, 1, 2))

    # # plot scatter plot of true vs predicted values for mid_z slice
    # import matplotlib.pyplot as plt
    # plt.scatter(x=images_aligned[:, :, :, :, mid_z].flatten(), y=spot_colours[:, :, :, :, mid_z].flatten())
    # plt.xlabel('True Values')
    # plt.ylabel('Predicted Values')
    # plt.title('True vs Predicted Values')
    # plt.show()
    #
    # # open napari viewer to check the images
    # import napari
    # v = napari.Viewer()
    # v.add_image(images_aligned, name='Aligned Images', colormap='red', blending='additive',
    #             contrast_limits=np.nanpercentile(images_aligned, [1, 99]))
    # v.add_image(images_disaligned, name='Disaligned Images', visible=False,
    #             contrast_limits=np.nanpercentile(images_disaligned, [1, 99]))
    # v.add_image(spot_colours, name='Spot Colours', colormap='green', blending='additive',
    #             contrast_limits=np.nanpercentile(spot_colours, [1, 99]))
    # v.dims.axis_labels = ['rounds', 'channels', 'y', 'x', 'z']
    # napari.run()

    # check that the spot colours are the same as the original images
    assert np.nanmax(np.abs(images_aligned - spot_colours)[:, :, :, :, mid_z]) < 0.01


def test_grid_sample():
    """
    Simple test of the grid sampling function that makes our heads hurt less.
    """
    # set up data
    brain = skimage.data.brain()
    # convert this from zyx to yxz to match our data
    brain = np.moveaxis(brain, 0, -1)
    im_sz = np.array(brain.shape)

    # set some points to sample
    rng = np.random.RandomState(0)
    random_points = rng.randint(0, im_sz - 1, (100, 3))
    # convert to torch tensors
    brain = torch.tensor(brain).float()
    random_points = torch.tensor(random_points).float()
    true_vals = brain[random_points[:, 0].long(), random_points[:, 1].long(), random_points[:, 2].long()].numpy()

    # sample the points using grid_sample

    # input will be of size [N, M, D, H, W] with
    # N = number of images in the batch (1 in this case)
    # M = number of channels in the image (1 in this case)
    # D = depth of the image (the y dimension, 256)
    # H = height of the image (the x dimension, 256)
    # W = width of the image (the z dimension, 10)

    # grid will be of size [N, D', H', W', 3] with
    # N = number of images in the batch (1 in this case)
    # D' = Depth of output grid (we use this as number of points to sample, 100 in this case)
    # H' = Height of output grid (1 in this case)
    # W' = width of the output grid (1 in this case)
    # 3 = 3D coordinates of the points to sample (z, x, y)

    # grid values should be between -1 and 1, so we need to scale the random points to be between -1 and 1
    random_points = 2 * random_points / (im_sz - 1) - 1
    random_points = random_points[:, [2, 1, 0]]  # convert from yxz to zxy
    random_points = random_points.float()  # convert to float
    predicted_vals = torch.nn.functional.grid_sample(
        input=brain[None, None, :, :, :], grid=random_points[None, :, None, None, :], mode="nearest"
    ).squeeze()
    # reshape the predicted values to be the same shape as the true values and turn into numpy
    predicted_vals = predicted_vals.numpy()

    # check that the predicted values are the same as the true values
    # import matplotlib.pyplot as plt
    # plt.scatter(x=true_vals, y=predicted_vals)
    # plt.xlabel('True Values')
    # plt.ylabel('Predicted Values')
    # plt.title('True vs Predicted Values')
    # plt.show()

    assert np.allclose(true_vals, predicted_vals, atol=1e-6)
