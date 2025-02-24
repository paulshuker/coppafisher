## PciSeq

For probabilistic cell typing with [pciSeq](https://github.com/acycliq/pciSeq), a DAPI background image and gene spot
positions must be exported.

??? "Filtered DAPI"

    To export the anchor's filtered DAPI stitched image

    ```py
    --8<-- "export_dapi_to_pciseq.py"
    ```

    The DAPI image can then be loaded into memory by

    ```py
    import numpy as np

    dapi_image = np.load("/path/to/dapi_image.npz")["arr_0"]
    ```

??? "Unfiltered DAPI"

    To export the anchor's unfiltered DAPI stitched image

    ```py
    --8<-- "export_unfiltered_dapi_to_pciseq.py"
    ```

    The DAPI image can then be loaded into memory by

    ```py
    import numpy as np

    dapi_image = np.load("/path/to/dapi_image.npz")["arr_0"]
    ```

Export gene reads into a compatible csv file by

```py
--8<-- "export_to_pciseq_0.py"
```

where method can be `"omp"`, `"prob"`, or `"anchor"` for each gene calling method. To set a score and/or intensity
minimum threshold:

```py
--8<-- "export_to_pciseq_1.py"
```

score_thresh and intensity_thresh must be numbers. Use the [Viewer](diagnostics.md#viewer) to help decide on thresholds.
intensity_thresh is set to `0.15` in the Viewer by default.

## Custom Images

There is a built-in tool to stitch then register additional images. When registered, the images are aligned with the
spot positions that are shown in the Viewer and exported for [pciSeq](#pciseq).

??? info "Registration Method"

    The registration uses the older method of sub volume registration (see
    [issue](https://github.com/paulshuker/coppafisher/issues/210) for a future optical flow enhancement).

### Extract the additional image(s)

The additional images must be extracted from the ND2 files. They are saved as tiff files. If you do not have ND2 input
files, you need to first manually convert them to tiff files.

```py
from coppafisher.custom_alignment import extract_raw
from coppafisher import Notebook

config_file = "/path/to/used/config.ini"
custom_nd2 = "/path/to/input/file.nd2"
output_dir = "/path/to/extract/directory/"

nb = Notebook("/path/to/notebook")
extract_raw(
    nb,
    config_file,
    save_dir=output_dir,
    read_dir=custom_nd2,
    use_tiles=nb.basic_info.use_tiles,
    use_channels=[9, 23],
    reverse_custom_z=False,
)
```

`use_channels` can be any valid channel(s) inside the custom image .nd2 file. This will also extract the anchor round in
the DAPI channel. You can reverse the z planes in the custom image by setting `reverse_custom_z` to `#!python True`.

??? note "Config File"

    The config file must be a valid configuration, like the one used during the experiment. Therefore, the `input_dir`
    must point to a real input directory.

### Stitch

The extracted raw anchor-DAPI images are stitched using coppafisher's [stitch](stitch.md) method. The custom image is
stitched by the same method separately. This then needs to be registered with the anchor-DAPI in the next step. Do this
for each custom image channel separately.

```py
from coppafisher.custom_alignment import fuse_custom_and_dapi

fused_custom_image, fused_anchor_image = fuse_custom_and_dapi(nb, output_dir, channel=0)
```

### Register

The custom fused image can be registered to the anchor DAPI fused image.

```py
from coppafisher.custom_alignment import register_custom_image

downsample_factor = 1  # Any natural number, `subvolume_size` along y and x is affected.
reg_parameters = {
    "registration_type": "subvolume",  # Can be "shift" or "subvolume".
    "subvolume_size": [8, 1024, 1024],
    "overlap": 0.1,  # Subvolume overlap.
    "r_threshold": 0.8,  # How good the subvolume shifts must be.
}

fused_custom_image = fused_custom_image[:, ::downsample_factor, ::downsample_factor]
fused_anchor_image = fused_anchor_image[:, ::downsample_factor, ::downsample_factor]

transform = register_custom_image(fused_anchor_image, fused_custom_image, reg_parameters, downsample_factor)
```

`subvolume_size` must be small enough for at least two subvolumes along every axis.

If downsample_factor is greater than 1, get the originally-sized images back by running the [stitch](#stitch) code
again.

### Apply transform and save results

Now apply the transform to the custom image and save the result as a .tif file.

```py
from coppafisher.custom_alignment import apply_transform

save_dir = "/path/to/output/directory"
apply_transform(fused_custom_image, transform, save_dir, name=f"custom_final_channel_{channel}.tif")
```

The custom image will be saved as the given name as `uint16`.

You can also save the fused_anchor_image which should have no transform applied to it.

```py
from coppafisher.custom_alignment import apply_transform

apply_transform(fused_anchor_image, None, save_dir, name="anchor_dapi.tif")
```
