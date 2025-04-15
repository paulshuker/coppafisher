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

Additional custom images can be aligned with coppafisher images and gene spots provided that you have a dapi channel (or
something similar to align with the anchor-DAPI image).

### Extract the additional image(s)

The additional images must be extracted from ND2 files. You will likely require the DAPI channel for best results. They
are saved as tiff files. If you do not have ND2 input files, you need to first manually convert them to tiff files.

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
    use_channels=[nb.basic_info.dapi_channel, 9, 23],
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
for each custom image channel separately. I suggest starting with the dapi channels first

```py
from coppafisher.custom_alignment import fuse_custom_and_dapi

fused_custom_dapi_image, fused_anchor_dapi_image = fuse_custom_and_dapi(nb, output_dir, channel=nb.basic_info.dapi_channel)
```

### Dapi Register

Alignment is done using a private package called `transform` maintained by Max Shinn (m.shinn@ucl.ac.uk). This allows
control over the type of transformation to apply based on their custom images. Install required dependencies

```terminal
python -m pip install --upgrade imageio-ffmpeg
```

Then in an empty directory, clone the package

```terminal
git clone --depth 1 https://github.com/mwshinn/transform.git
cd transform
ipython
```

Then start the interactive alignment process

```py
import transform.gui
from transform.base import TranslateRotate

round_transform = transform.gui.alignment_gui(
    fused_custom_dapi_image, fused_anchor_dapi_image, transform_type=TranslateRotate
)
```

Press `Add new point` and click twice to place two corresponding points. Do this a few times, preferably in various z
planes too. Press `Perform transform` to see the resulting transform. Once you are happy with the result, close the
napari window.

??? info "Type of Transform"

    You can change the type of transform you wish to find, please see the transfrom
    [readme](https://github.com/mwshinn/transform/blob/master/README.md) for details.

    For example, you could use the more robust transform type of `#!python TranslateRotateRescale`. It requires four
    points in every corner of the image on both edges of the z stack for best results.

??? tip "Save and Load Transforms"

    Every transform can be saved and reloaded at a later point. You just need to save the text representation of the
    transform which can be found by

    ```py
    str(round_transform)
    ```

    You can save it using Python

    ```py
    with open("/path/to/saved/transform.txt", "w") as file:
        file.write(str(round_transform))
    ```

    It can be reloaded by

    ```py
    from transform.base import *

    with open("/path/to/saved/transform.txt", "r") as file:
        round_transform = eval("\n".join(file.readlines()))
    ```

    Do not run `#!python eval` on stranger's code (it could be malicious)!

You can now apply the resulting transform to the custom dapi image and save the result as a `.tif` file

```py
import numpy as np
import tifffile

fused_custom_dapi_image_transformed = round_transform.transform_image(
    fused_custom_dapi_image, relative=fused_anchor_dapi_image.shape, force_size=True, labels=True
)
tifffile.imwrite("/path/to/saved/custom_dapi_image_transformed.tif", fused_custom_dapi_image_transformed)
del fused_custom_dapi_image_transformed
del fused_anchor_dapi_image
```

### Non-Dapi Register

For a non-dapi custom image channel `c`, it is recommended to find a specific transform to move to the dapi channel for
best registration. To do this, first find a transform to move into the dapi custom image's frame

```py
from coppafisher.custom_alignment import fuse_custom_and_dapi
import transform.gui
from transform.base import TranslateRotate

fused_custom_channel_image, _ = fuse_custom_and_dapi(nb, output_dir, channel=c)

channel_transform = transform.gui.alignment_gui(
    fused_custom_channel_image, fused_custom_dapi_image, transform_type=TranslateRotate
)
```

Now save the fully registered channel image

```py
import tifffile

fused_custom_channel_image_transformed = (channel_transform + round_transform).transform_image(
    fused_custom_channel_image, relative=fused_custom_channel_image.shape, force_size=True, labels=True
)
tifffile.imwrite(f"/path/to/saved/custom_channel_{c}_image_transformed.tif", fused_custom_channel_image_transformed)
del fused_custom_channel_image_transformed
del fused_custom_channel_image
```
