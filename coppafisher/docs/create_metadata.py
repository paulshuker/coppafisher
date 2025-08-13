import json
import os

# CHANGE THESE PARAMETERS:
output_dir = "/path/to/output/directory/"
n_tiles = 2
n_rounds = 7
n_total_channels = 28
n_y = 2048
n_z = 50
pixel_size_xy = 0.3
pixel_size_z = 0.9
tile_origins_yx = [[1, 0], [0, 0]]

if __name__ == "__main__":
    metadata = {
        "n_tiles": n_tiles,
        "n_rounds": n_rounds,
        "n_channels": n_total_channels,
        "tile_sz": n_y,  # or n_x
        "pixel_size_xy": pixel_size_xy,
        "pixel_size_z": pixel_size_z,
        "tile_centre": [n_y / 2, n_y / 2, n_z / 2],
        "tilepos_yx": tile_origins_yx,
        "tilepos_yx_nd2": list(reversed(tile_origins_yx)),
        "channel_camera": [1] * n_total_channels,
        "channel_laser": [1] * n_total_channels,
        "nz": n_z,
    }
    file_path = os.path.join(output_dir, "metadata.json")
    with open(file_path, "w") as f:
        json.dump(metadata, f, indent=4)
