# Find Spots

Spots are detected as local intensity maxima on the filtered images to create 3D point clouds. Spot positions are used
later in [register](register.md) and [call spots](call_spots.md).

## 1: Spot Detection

For each tile/round/channel filtered image ($I_{trcxyz}$), a 3D point cloud is created out of all points where

$$
I_{trcxyz} > \text{threshold}_{trc}
$$

where `threshold` (typically `0.2`) is given in the find spots config.

## 2: Remove Duplicates

Some of these points will be nearby or adjacent, representing the same spot multiple times. To deal with this, points
are removed. For each point, an ellipsoid region of x/y radius `radius_xy` (typically $5$) and z radius `radius_z`
(typically $2$) is considered. If one or more points are found within the point's ellipsoid region, then only the point
with the greatest intensity is kept. If a point is isolated, it is kept. The process is repeated for all points.

If too few spots are found for a tile/round/channel image, then a warning and/or error is raised to the user.

## Diagnostics

### Detected Spots

A find spots viewer will help to understand the config parameters. Move sliders to adjust the find spots parameters and
see the spot detection results. You can select any tile/round/channel combination. By default, the first tile's anchor
round/channel is shown. To display the viewer, in the terminal

```terminal
python -m coppafisher -fs /path/to/notebook
```

and open the [link](http://127.0.0.1:8050/) shown in the terminal on a modern browser to show and interact with the
viewer. Press Ctrl + C in the terminal to close the viewer.

Spot detection is most crucial for the anchor round/channel images. So, it is recommended to configure find spots
parameters based on one of these images.
