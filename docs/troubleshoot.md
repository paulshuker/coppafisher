## Pipeline crash

If the coppafish pipeline is crashing, first read the error message. If there is a suggestion about how to fix the
issue in the config, try changing the config variable and running the pipeline again. If the suggestion does not make 
sense to you, feel free to reach out to a developer for help or 
[create an issue](https://github.com/paulshuker/coppafish/issues/new?assignees=&labels=&projects=&template=bug.md&title=) 
on GitHub!

## Cannot open napari issues

If napari fails to open and you see an error such as

``` bash
WARNING: composeAndFlush: makeCurrent() failed
```

when trying to open the Viewer or RegistrationViewer, here are a few suggestions that might fix the issue:

* In the conda environment, run `#!terminal conda install -c conda-forge libstdcxx-ng`
* In the conda environment, run `#!terminal conda install -c conda-forge libffi`.

## Memory crash at OMP

Try reducing `subset_pixels` in the `[omp]` config section. This will cause OMP to compute on fewer pixels at time. It 
has a minimal effect on compute times, but will lower the RAM (vRAM) usage. By default, the subset pixels number is 
found on the fly based on the PC's hardware, you can check this value by looking at the pipeline.log file and searching 
for `subset_pixels`.
