## Pipeline crash

If the coppafisher pipeline is crashing, first read the error message. If there is a suggestion about how to fix the
issue in the config, try changing the config variable and run the pipeline again. If the suggestion does not make sense
to you, feel free to reach out to the developer(s) for help or
[create an issue](https://github.com/paulshuker/coppafisher/issues/new?assignees=&labels=&projects=&template=bug.md&title=)
on GitHub!

## Joblib related crashes

If you find any mysterious crash during filter with an error message that traces back to the `joblib` python package,
you can disable joblib entirely by adding

```ini
[filter]
num_cores = 1
```

## Find spots does not find enough spots

This is a common issue that happens. It is usually caused by the default automatic threshold parameters not fitting your
particular dataset due to a difference in Signal-to-Noise Ratio. If your pipeline stops after find spots due to this
reason, follow these steps:

1. Open the find spots viewer, explained under the [Detected Spots](find_spots.md#detected-spots) section.

2. Here, you can tweak the multiplier and percentile and see how this affects the spot detections. You can choose to
have a single multiplier and percentile shared for all images or you can choose different values for different channels.

3. Once you are happy with the parameter tweaks, add the new values into the dataset's configuration file, e.g.

```ini
[find_spots]
auto_thresh_multipliers = 10
auto_thresh_percentiles = 25
```

for the same values for all channels, or something like

```ini
[find_spots]
auto_thresh_multipliers = 10, 35, 5, 40
auto_thresh_percentiles = 25, 20, 25, 25
```

for each sequence channel. By default, `auto_thresh_multipliers = 180` and `auto_thresh_percentiles = 5`.

4. Delete the notebook page `find_spots`, explained [here](advanced_usage.md#delete-notebook-page).

5. [Run the pipeline](basic_usage.md#running) again.

## Memory crash at OMP

Try reducing `subset_pixels` in the OMP config. This will cause OMP to compute on fewer pixels at time. It has a minimal
effect on compute times, but will lower the RAM/VRAM usage. By default, the subset pixel number is found on the fly
based on the PC's hardware, you can see this by looking at the pipeline.log file and searching for `subset_pixels`.
