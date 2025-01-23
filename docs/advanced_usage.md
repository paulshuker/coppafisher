## Change configuration

Each coppafisher section is saved as separate notebook page(s). To change the config variables and re-run the
coppafisher pipeline, you can delete the notebook and all output directory files and re-run again. But, if you only
wished to re-run starting from an intermediate stage, you can delete all subsequent stages and output files. To see what
valid stages of coppafisher you can re-run starting from, in chronological order, in the python terminal

```py
--8<-- "compatibility_tracker_stage_names.py"
```

As an example, if you wished to know how to start from the stage "find_spots" again

```py
--8<-- "compatibility_tracker_start_from.py"
```

and follow the instructions given. Then, you are safe to change the configuration for all sections after find spots. If
you are told to delete notebook page(s), see [here](#delete-notebook-page).

## Skipping bad microscope images

You may have one or more images that are taken which are corrupted, empty, or not as bright as expected. When this
happens, you can tell coppafisher to run without these images. To do this, specify each tile (`t`), round (`r`), channel
(`c`) image by going to your custom config file and add the line

```ini
[basic_info]
; Keep other options.
bad_trc = t1, r1, c1, t2, r2, c2, ...
```

under the `basic_info` section. Each set of brackets represents one image to ignore. This allows for meaningful
results to be salvaged from an incomplete tile.

## Export for pciSeq

For probabilistic cell typing with [pciSeq](https://github.com/acycliq/pciSeq), you can export gene reads into a
compatible csv file by

```py
--8<-- "export_to_pciseq_0.py"
```

where method can be `"omp"`, `"prob"`, or `"anchor"` for each gene calling method. To set a score and/or intensity
minimum threshold:

```py
--8<-- "export_to_pciseq_1.py"
```

score_thresh and intensity_thresh must be numbers. Use the [Viewer](diagnostics.md#viewer) to help decide on thresholds.

## Additional Image Registration and Stitching

There is a built-in tool to stitch then register additional images, (typically these are IF images, so it will called as
such from now on). Registration uses the older method of sub volume registration (see
[issue](https://github.com/paulshuker/coppafisher/issues/210) for a future optical flow enhancement).

### Extract the additional image(s)

The additional images must be extracted from the ND2 files. They are saved as tiff files. If you do not have ND2 input
files, you need to manually convert them to tiff files.

```py
from coppafisher.if_alignment import extract_raw
from coppafisher import Notebook

config_file = "/path/to/used/config.ini"
if_nd2_dir = "/path/to/input/file.nd2"
if_output_dir = "/path/to/output/directory"

nb = Notebook("/path/to/notebook")
extract_raw(nb, config_file, save_dir=if_output_dir, read_dir=if_nd2_dir, use_tiles=nb.basic_info.use_tiles, use_channels=[0,9,18,23])
```

where use_channels can be any number of channels.

### Stitch

Generate a globally-stitched image based on the coppafisher stitching results from the notebook.

```py
from coppafisher.if_alignment import stitch_if_and_dapi

stitch_if_and_dapi(nb, if_output_dir, use_channels=[0,9,18,23])
```

### Register

```py
from coppafisher.if_alignment import register_if

transform = register_if(seq_im, if_im, downsample_factor_yx=4, transform_save_dir=transform_save_dir, reg_parameters = reg_parameters)
```

## Create a background process

Large datasets can have a long compute time (in the order of days). It is recommended to run these by setting them up as
a background process. It is not recommended to run multiple large datasets at once since they will be fighting for
memory, CPU, and disk I/O resources. Running in the background depends on the operating system:

### Linux

Start a background process from a terminal with the coppafisher conda environment activated

```bash
nohup python3 -m coppafisher /path/to/config.ini &
```

the background process will run, even if the terminal is now closed. Follow its progress by

```bash
tail -f nohup.out
```

press Ctrl + C to stop following. The process can be killed by finding it after running a command like `htop`,
highlighting it, press F9, then Enter to kill it. Press q to exit the `htop` view.

### Windows

Open command prompt or powershell, run the command

```powershell
start /b python -m C:\path\to\config.ini
```

Try to keep the command prompt open to watch the progress. Do not log out or shutdown the PC while the process is still
running.

## Delete notebook page

To remove a notebook page, in the python terminal

```py
--8<-- "nb_delete_page_0.py"
```

For example, to remove the stitch page

```py
--8<-- "nb_delete_page_1.py"
```

Any page's added after stitch are warned about. It is recommended to delete these pages as well by typing `y` then
pressing enter.

## Disable GPU

You can force coppafisher to run on the CPU only by adding to the config file

```ini
[filter]
force_cpu = true

[omp]
force_cpu = true
```

Filter and OMP are the only stages that leverage the GPU.

## Email notification

To be emailed when the pipeline crashes or finishes, under section `[notifications]` in the config, add the variable
`email_me` with your email address. You must have a sender email with SMTP support, this email's credentials must be
given in `[notifications]` under the variables `sender_email` and `sender_email_password`. The email may be flagged as
junk or not be sent altogether depending on the email address you are sending to. This has only been tested for an
"outlook.com" Microsoft email.

## Generate gene codes

Generate gene codes automatically in the python terminal by

```py
--8<-- "generate_gene_codes.py"
```

where `n_gene_codes` is the number of gene codes desired, `n_rounds` is the number of sequencing rounds, and
`n_channels` is the number of channels. An error is thrown if the number of unique gene codes desired is impossible to
create. Each channel is labelled 0, 1, 2, ... and `codes` is a dictionary. Each gene code generated can be accessed. For
example, to access the first gene code: `codes["gene_0"]`.

## Retrieve the Notebook config

Every notebook page has associated config section(s) saved to disk. You can look at each notebook page's associated
config section(s). For example, to see the associated config section(s) for the filter page, in the python terminal

```py
--8<-- "retrieve_notebook_config.py"
```
