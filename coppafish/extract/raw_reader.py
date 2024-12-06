import numbers
from typing import Any, Tuple

import dask
import numpy as np
import numpy_indexed

from ..setup.notebook_page import NotebookPage


class RawReader:
    """
    The general raw reader class.

    Every supported raw extension type inherits the RawReader and creates their own way of reading the raw images as
    ndarrays for coppafish extraction.
    """

    def read(
        self, nbp_basic: NotebookPage, nbp_file: NotebookPage, tile: int, round: int, channels: list[int]
    ) -> np.ndarray:
        """
        Base function to read raw files for the given tile, round for the specific channels.

        Args:
            nbp_basic (NotebookPage): `basic_info` notebook page.
            nbp_file (NotebookPage): `file_names` notebook page.
            tile (int): tile index.
            round (int): round index.
            channels (list of int): the channels to gather.

        Returns:
            ((tuple of length `len(channels)`) containing `(im_y x im_x x im_z) ndarrays`): images. The tile/round
                images. Can be any datatype, but we work with uint16 raw files. They are later converted to floating
                point numbers when filtering the images.
        """
        assert type(nbp_basic) is NotebookPage
        assert type(nbp_file) is NotebookPage
        assert type(tile) is int
        assert type(round) is int
        assert type(channels) is list
        assert len(channels) > 0
        assert all([type(c) is int for c in channels])

    def load_as_dask(
        self, nbp_basic: NotebookPage, nbp_file: NotebookPage, round: int
    ) -> Tuple[dask.array.Array, dict[str, Any] | None]:
        """
        Memmap load the round dask array containing images for all tiles and channels in the given round. The function
        must be completed by classes that inherit from RawReader.

        Args:
            nbp_basic (NotebookPage): `basic_info` notebook page.
            nbp_file (NotebookPage): `file_names` notebook page.
            round (int): round index. The anchor will be assumed to be the last round.

        Returns:
            Tuple containing:
                - (dask Array): round_dask_array. Array with indices in order `fov`, `channel`, `y`, `x`, `z`.
                - (dict[str, any] or none): all_metadata. Dictionary of all found metadata in ND2 file. None if not
                    applicable.
        """

        assert type(nbp_basic) is NotebookPage
        assert type(nbp_file) is NotebookPage
        assert type(round) is int
        assert round >= 0

        n_tiles, tile_sz, nz = nbp_basic.n_tiles, nbp_basic.tile_sz, nbp_basic.nz
        if nbp_basic.channel_laser is not None:
            channel_laser = list(set(nbp_basic.channel_laser))
        else:
            channel_laser = []
        channel_laser.sort()
        if nbp_basic.channel_camera is not None:
            channel_cam = list(set(nbp_basic.channel_camera))
        else:
            channel_cam = []
        channel_cam.sort()

        # If the camera argument is left blank in the notebook, this just means we only have one camera
        # Not sure how this applies to lasers
        n_lasers, n_cams = max(1, len(np.unique(channel_laser))), max(1, len(np.unique(channel_cam)))

        all_metadata = None

        if nbp_file.raw_extension != "jobs":
            if nbp_basic.use_anchor:
                # always have anchor as first round after imaging rounds
                round_files = nbp_file.round + [nbp_file.anchor]
            else:
                round_files = nbp_file.round
            round_file = os.path.join(nbp_file.input_dir, round_files[round])
            if nbp_file.raw_extension == ".nd2":
                round_dask_array = nd2.load(round_file + nbp_file.raw_extension)
                all_metadata = nd2.get_all_metadata(round_file + nbp_file.raw_extension)
            elif nbp_file.raw_extension == ".npy":
                round_dask_array = dask.array.from_npy_stack(round_file)
        else:
            # TODO: Combine all metadata from the jobs nd2 files for each round.
            # Now deal with the case where files are split by laser
            if nbp_basic.use_anchor:
                # always have anchor as first round after imaging rounds
                round_files = nbp_file.round + [nbp_file.anchor]
            else:
                round_files = nbp_file.round

            round_laser_dask_array = []
            # Deal with non anchor round first as this follows a different format to anchor round
            if round != nbp_basic.anchor_round:

                for t in tqdm(range(n_tiles), desc="Loading tiles in dask array"):
                    # Get all the files of a given tiles (should be 7)
                    tile_files = round_files[round][t * n_lasers : (t + 1) * n_lasers]
                    tile_dask_array = []

                    for f in tile_files:
                        laser_file = os.path.join(nbp_file.input_dir, f + ".nd2")
                        tile_dask_array.append(nd2.load(laser_file))

                    tile_da = dask.array.concatenate(tile_dask_array, axis=-1)  # concatenate on the laser axis
                    tile_da = dask.array.swapaxes(tile_da, -1, 0)  # we need 'channel', 'z', 'y','x'

                    round_laser_dask_array.append(tile_da)

            # If we're dealing with the anchor round, we only need channel 0 (DAPI) and anchor
            # The rest of the array is padded with zeros

            else:

                anchor_laser_index = channel_laser.index(nbp_basic.channel_laser[nbp_basic.anchor_channel])
                dapi_laser_index = channel_laser.index(nbp_basic.channel_laser[nbp_basic.dapi_channel])

                anchor_files = nbp_file.anchor

                for t in tqdm.tqdm(range(n_tiles), desc="Loading tiles in dask array"):
                    # Get all the files of a given tiles (should be 7)
                    tile_files = anchor_files[t * n_lasers : (t + 1) * n_lasers]
                    tile_dask_array = []
                    latest_shape = (nz + 1, tile_sz, tile_sz, n_cams)

                    for f_id, f in enumerate(tile_files):
                        if f_id == anchor_laser_index or f_id == dapi_laser_index:
                            laser_file = os.path.join(nbp_file.input_dir, f + ".nd2")
                            new_dask_array = nd2.load(laser_file)
                            latest_shape = new_dask_array.shape
                            tile_dask_array.append(new_dask_array)
                            del new_dask_array
                        else:
                            tile_dask_array.append(dask.array.zeros(latest_shape, dtype=np.uint16))
                            # TODO find a better fix for nz. here it is different because of basic_info use_z
                            # Ideally it should have the same shape as the array for dapi

                    tile_da = dask.array.concatenate(tile_dask_array, axis=-1)  # concatenate on the laser axis
                    tile_da = dask.array.swapaxes(tile_da, -1, 0)  # we need 'channel', 'z', 'y','x'

                    round_laser_dask_array.append(tile_da)

                # now concatenate dask arrays
            round_dask_array = dask.array.stack(round_laser_dask_array, axis=0)

        return round_dask_array, all_metadata

    def get_tile_raw_index(self, tile_index: int, tile_pos_yx_raw: np.ndarray, tile_pos_yx: np.ndarray) -> int:
        """
        Convert the given, coppafish tile index into the raw tile index. This is required since coppafish uses a
        different tile ordering system to Nikon's ND2 files which we use as raw input.

        Args:
            tile_index (int): the coppafish tile index to convert.
            tile_pos_yx_raw (`(n_tiles x 2) ndarray[int]`): tile_pos_yx_raw[i] contains the y and x position of tile at
                index `i`. i = 0 refers to y, x = 0, 0. i = 1 refers to y, x = 0, 1 if x tile count > 1.
            tile_pos_yx (`(n_tiles x 2) ndarray[int]`): tile_pos_yx[i] contains the y and x position of tile at index
                `i`. i = 0 refers to y, x = [maxY, maxX]. i = 1 refers to y, x = [maxY, maxX - 1] if x tile count > 1.

        Returns:
            (int): the raw tile index.
        """
        if isinstance(tile_index, numbers.Number):
            tile_index = [tile_index]
        # As npy and nd2 have different coordinate systems, we need to convert tile_pos_yx_npy to nd2 tile coordinates
        tile_pos_yx = np.max(tile_pos_yx, axis=0) - tile_pos_yx
        # TODO: Remove the obscure dependency for a line.
        nd2_index = numpy_indexed.indices(tile_pos_yx_raw, tile_pos_yx[tile_index]).tolist()
        if len(nd2_index) == 1:
            nd2_index = nd2_index[0]

        return nd2_index

    def get_channel_laser(self, nbp_basic: NotebookPage) -> list[int]:
        """
        Get a list of unique wavelengths in nm for the lasers used.
        """
        assert type(nbp_basic) is NotebookPage

        if nbp_basic.channel_laser is not None:
            channel_laser = list(set(nbp_basic.channel_laser))
        else:
            channel_laser = []
        channel_laser.sort()

        return channel_laser

    def get_channel_cam(self, nbp_basic: NotebookPage) -> list[int]:
        """
        Get a list of unique wavelengths in nm for the camera in all channels.
        """
        assert type(nbp_basic) is NotebookPage

        if nbp_basic.channel_camera is not None:
            channel_camera = list(set(nbp_basic.channel_camera))
        else:
            channel_camera = []
        channel_camera.sort()

        return channel_camera
