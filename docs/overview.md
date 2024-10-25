The coppafish pipeline is separated into distinct stages. Some of these are for image pre-processing 
([extract](#extract), [filter](#filter)), image alignment ([register](#register), [stitch](#stitch)) and spot 
detection/gene calling ([find spots](#find-spots), [call spots](#call-spots), 
[orthogonal matching pursuit](#orthogonal-matching-pursuit)). Below, each stage is given in chronological order.

## Extract

Save all raw data again at the `tile_dir` in the `extract` config section. Coppafish does this for:

* file compression.
* saving raw data in a consistent format.
* faster data retrieval. The default file type is using [zarr](https://zarr.readthedocs.io/) arrays, but coppafish also
supports saving as uncompressed numpy arrays by setting `file_type` to `.npy` in the extract config section.

Extract also saves metadata inside of the `tile_dir` directory if the raw files are ND2 format.

## Filter

All images are filtered to help minimise scattering of light (bright points will appear as cones initially, hence the
name "Point Spread Function") and emphasise spots. A given point spread function is used to Wiener deconvolve the 
images.

After filtering is applied, the images are saved to the notebook as `float16`.

## Find spots

Point clouds (a series of spot x, y, and z locations) are generated for each filtered image. These are found by
detecting local maxima in image intensity around the rough spot size (specified by config variables `radius_xy` and
`radius_z` in the `find_spots` section). If two local maxima are the same value and in the same spot region, then one
is chosen at random. Warnings and errors are raised if there are too few spots detected in a round/channel, these can
be customised, see `find_spots` section in the
<a href="https://github.com/paulshuker/coppafish/blob/HEAD/coppafish/setup/settings.default.ini" target="_blank">
config</a> default file for variable names.

## Register

## Stitch

## Call spots

## Orthogonal Matching Pursuit

Orthogonal Matching Pursuit (OMP) is the most sophisticated gene calling method used by coppafish, allowing for
overlapping genes to be detected. It is an iterative,
<a href="https://en.wikipedia.org/wiki/Greedy_algorithm" target="_blank">greedy algorithm</a> that runs on individual
pixels of the images. At each OMP iteration, a new gene is assigned to the pixel. OMP is also self-correcting.
"Orthogonal" refers to how OMP will re-compute its gene contributions (their coefficients) after every iteration by 
least squares. Background genes[^1] are considered valid genes in OMP. The iterations stop if:

* `max_genes` in the `omp` config section is reached.
* assigning the next best gene to the pixel does not have a dot product score above `dp_thresh` in the `omp` config. 
The dot product score is a dot product of the residual pixel intensity in every sequencing round/channel (known as its 
colour) with the normalised bled codes (see [call spots](#call-spots)).

<!-- TODO: Should expand more on the OMP gene scoring here -->
Every coefficient pixel is scored by a weighted average around a small local region of the spot where the spot is 
expressed most strongly. The coefficients are weighted with the mean spot intensity normalised to have a maximum of 1. 
The mean spot is computed on tile `nb.basic_info.use_tiles[0]` by taking the average of many well-isolated spots. The 
scoring is controlled by config parameters `shape_isolation_distance_yx`, `shape_isolation_distance_z`, 
`shape_coefficient_threshold` and `shape_sign_thresh`. Low scores are deleted by OMP when they are below the 
`score_threshold`.

Since OMP is sensitive to the many steps before, it can be difficult to optimise. This is partly why 
[call spots](#call-spots) is part of the gene calling pipeline, known for its simpler and more intuitive method. A good 
sanity check is to see if OMP and call spots have similar gene reads. But, you should expect more gene calls made by 
OMP.

## Runtime

For an estimate of your pipeline runtime, in the Python terminal:
```python
from coppafish.utils import estimate_runtime

estimate_runtime()
```
then type in the relevant information when prompted[^2].


[^1]:
    Background genes refer to constant pixel intensity across all sequencing rounds in one channel. This is an
    indicator of an anomalous fluorescing feature that is not a spot. No spot codes are made to be the same channel in
    all rounds so they are not mistaken with background fluorescence.
[^2]:
    All time estimations are made using an Intel i9-13900K @ 5.500GHz, NVIDIA RTX 4070Ti Super, and NVMe local SSD. 
    Raw, ND2 files were saved on a server with read speed of ~200MB/s.
