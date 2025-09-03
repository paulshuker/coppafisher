## Input data

Coppafisher requires raw, `uint16` microscope images, metadata, and a configuration file. We currently only support raw
data in ND2, JOBs, numpy, or tif format. If your data is not already in one of these formats, we recommend configuring
your data into [numpy](#numpy) format. The [tif](#tif) file format is also explained below.

There must be an anchor round. There must be an anchor channel (this can be a sequencing channel). There must be a dapi
channel in every sequencing round and the anchor round. The tiles must have at least four z planes. Use a number of z
planes that is a multiple of two.

??? info "Tile Indexing Conventions"

    Input tiles can be indexed differently to coppafisher. You can use [this](diagnostics.md#view-tile-indexing) diagnostic.

### Numpy

Each round is separated between directories. Label sequencing round directories `0`, `1`, etc. We recommend using
[dask](https://docs.dask.org), this is installed in your coppafisher environment by default. The code to save input
data:

```python
import os
import dask.array

raw_path = "/path/to/raw/data"
dask_chunks = (1, n_total_channels, n_y, n_x, n_z)
for r in range(n_seq_rounds):
    save_path = os.path.join(raw_path, f"{r}")
    image_dask = dask.array.from_array(seq_image_tiles[r], chunks=dask_chunks)
    dask.array.to_npy_stack(save_path, image_dask)

# Anchor round
save_path = os.path.join(raw_path, "anchor")
image_dask = dask.array.from_array(anchor_image, chunks=dask_chunks)
dask.array.to_npy_stack(save_path, image_dask)
```

where `n_...` variables represent counts (integers), `seq_image_tiles` is a numpy array of shape
`(n_seq_rounds, n_tiles, n_total_channels, n_y, n_x, n_z)`, while `anchor_image` is a numpy array of shape
`(n_tiles, n_total_channels, n_y, n_x, n_z)`. Note that `n_y` must equal `n_x`.


### Tif

Every round (anchor included) must be a .tif file located inside of the `input_dir`. They must have the shape
`(n_tiles * n_total_channels * n_z, n_y, n_x)`. The first axis is flattened such that the first n_total_channels are
tile 0 and z plane 0 on each channel, then the next n_total_channels are tile 0 and z plane 1 on each channel. Then
after n_z z planes the next n_total_channels are tile 1 and z plane 0 on each channel etc...

### Metadata

The metadata file required for [numpy](#numpy) and [tif](#tif) input formats. It must be saved in the same location as
the raw input files. This can be done using Python:

```python
--8<-- "create_metadata.py"
```

`n_tiles` must be the total number of tiles inside of the raw inputted files (even if you only plan on selecting a
subset of them). Similarly, `n_total_channels` must be the total number of channels in the inputted raw files.

`pixel_size_xy` is the size of a pixel along the y/x axes in microns. `pixel_size_z` is the size of a pixel along the z
axis in microns. `n_y` is the number of pixels along y/x for a single tile. `n_z` is the number of pixels along z for a
single tile. `tile_origins_yx` is a list of lists which tells coppafisher where each tile is relative to one another.
For example, a 2x2 of tiles going around clockwise starting from the top-left would be
`#!python tile_origins_yx = [[0, 0], [0, 1], [1, 1], [1, 0]]`.

### Code book

A code book is a `.txt` file that tells coppafisher the gene codes for each gene. Each digit is the dye index for each
sequencing round. An example of a four gene code book is

```text
gene_0 0123012
gene_1 1230123
gene_2 2301230
gene_3 3012301
```

the names (`gene_0`, `gene_1`, ...) can be changed. Do not assign any genes a constant gene code like `0000000`. To
learn how the codes can be generated, see [advanced usage](advanced_usage.md#generate-gene-codes). For details on how
the codes are best generated, see `reed_solomon_codes` in the
[source code](https://github.com/paulshuker/coppafisher/blob/HEAD/coppafisher/utils/base.py). See
[Wikipedia](https://en.wikipedia.org/wiki/Reed%E2%80%93Solomon_error_correction) for algorithmic details on how gene
codes are best selected.

### Configuration

There are configuration variables used throughout the coppafisher pipeline. Most of these have reasonable default
values, but some must be set by the user and you may wish to tweak other values for better performance. Save the config
text file, like `dataset_name.ini`. The config file should contain, at the minimum:

```ini
[file_names]
; MUST SPECIFY input_dir, output_dir, tile_dir, code_book.
input_dir =
output_dir =
tile_dir =
code_book =
; This can be .npy, .tif, .nd2 or jobs.
raw_extension = .npy
; The names of the ND2 files (excluding the file extension above).
round = round0, round1, round2, round3, round4, round5, round6
anchor = anchor
; Optional, leave blank if you do not have a fluorescent bead file.
fluorescent_bead_path =

[basic_info]
; The names of the dyes given, must match the number of dyes used in the gene codebook.
dye_names = dye_0, dye_1, dye_2, dye_3
; Optional, leave blank to run on all tiles.
use_tiles =
; Round indices (starting from 0) located in the input files.
use_rounds = 0, 1, 2, 3, 4, 5, 6
; Channel indices (starting from 0) located in the input files.
use_channels = 5, 9, 10, 14, 15, 18, 19, 23, 27
; Optional, leave blank to run on all z planes.
use_z =
; The index of the anchor round.
anchor_round = 7
; The index of the anchor channel.
anchor_channel = 1
; The index of the dapi channel.
dapi_channel = 0

[stitch]
; The percentage overlap between adjacent tiles.
expected_overlap = 0.1

[call_spots]
target_values = 1, 1, 1, 1
d_max = 0, 1, 2, 3
```

`raw_extension` is `.npy` for [numpy](#numpy) input, `.tif` for [tif](#tif) input, `.nd2` for nd2 input, and `jobs` for
JOBs input.

`tile_dir` is the tile directory, where extract images are saved to, it should be empty before running coppafisher.
`output_dir` is where the notebook and PDF diagnostics are saved, it should also be blank before running. More details
about every config variable can be found at
<a href="https://github.com/paulshuker/coppafisher/blob/HEAD/coppafisher/setup/default.ini" target="_blank">
`coppafisher/setup/default.ini`</a> in the source code.

`target_values` and `d_max` must both have `n_seq_channels` numbers, one for each channel. See
[call spots](call_spots.md#4-round-and-channel-normalisation) for details on how to set the values. If you are unsure,
set target_values to all ones and d_max to the brightest channel in each dye.

??? info "Unique anchor raw file indices"

    If your anchor raw file has unique channel locations compared to the sequencing raw files, set
    `raw_anchor_channel_indices` under the `file_names` section in the config. Go to
    <a href="https://github.com/paulshuker/coppafisher/blob/HEAD/coppafisher/setup/default.ini" target="_blank">
    `coppafisher/setup/default.ini`</a> and search for `raw_anchor_channel_indices` for a description and usage.

## Running

Coppafisher must be run with a [configuration](#configuration) file. In the command line

```terminal
python3 -m coppafisher /path/to/config.ini
```

Or programmatically, using a python script

```py
--8<-- "run_pipeline_0.py"
```

which can then be run from the command line

```bash
python3 coppafisher_script_name.py
```

## Runtime

For an estimate of your pipeline runtime[^1], in the Python terminal:

```python
--8<-- "estimate_runtime.py"
```

then type in the relevant information when prompted.


[^1]:
    All time estimations are made using an Intel i9-13900K @ 5.500GHz, NVIDIA RTX 4070Ti Super (optional), and NVMe
    local SSD. Raw, ND2 input files were saved on a server with read speed of ~200 MB/s.
