Almost all versions of coppafisher will automatically warn you about any incompatibilities when the pipeline is run on a
new version. Below contains specific migration guides for special cases.

## Coppafisher $\leq$ 1.5 Migration

The extraction directory (labelled `tile_dir` under the config) is deprecated. Update the directory by

```python
--8<-- "update_tile_dir.py"
```

The notebook from versions <= 1.5.0 will open as normal. The data can be zipped at any time to be the same as versions >
1.5.0 by

```python
--8<-- "zip_nb.py"
```

You may be prompted to install the 7-zip CLI if it is not found.
