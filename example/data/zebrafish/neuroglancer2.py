import daisy
import neuroglancer
import numpy as np
import zarr

import logging

logger = logging.getLogger(__name__)


def add_layer(context, array, name, visible=True, **kwargs):
    array_dims = len(array.shape)
    voxel_size = array.voxel_size
    attrs = {
        2: {"names": ["y", "x"], "units": "nm", "scales": voxel_size},
        3: {"names": ["z", "y", "x"], "units": "nm", "scales": voxel_size},
        4: {
            "names": ["c^", "z", "y", "x"],
            "units": ["", "nm", "nm", "nm"],
            "scales": [1, *voxel_size],
        },
    }
    dimensions = neuroglancer.CoordinateSpace(**attrs[array_dims])
    offset = np.array((0,) * (array_dims - 3) + array.roi.get_offset())
    offset = offset // attrs[array_dims]["scales"]
    # if len(offset) == 2:
    #     offset = (0,) + tuple(offset)

    d = np.asarray(array.data)
    if array.data.dtype == np.dtype(bool):
        array.data = np.array(d, dtype=np.float32)

    channels = ",".join(
        [
            f"toNormalized(getDataValue({i}))" if i < array.shape[0] else "0"
            for i in range(3)
        ]
    )
    shader_4d = (
        """
void main() {
  emitRGB(vec3(%s));
}
"""
        % channels
    )
    shader_3d = """
void main () {
  emitGrayscale(toNormalized(getDataValue()));
}"""

    layer = neuroglancer.LocalVolume(
        data=array.data, dimensions=dimensions, voxel_offset=tuple(offset)
    )

    if array.data.dtype == np.dtype(np.uint64):
        context.layers.append(name=name, layer=layer, visible=visible)
    else:
        context.layers.append(
            name=name,
            layer=layer,
            visible=visible,
            shader=shader_4d if array_dims == 4 else shader_3d,
            **kwargs,
        )


def get_volumes(h5_file, path):
    datasets = []
    try:
        for key in h5_file.get(path, {}).keys():
            datasets += get_volumes(h5_file, f"{path}/{key}")
        return datasets
    except AttributeError:
        return [path]


def add_snapshot(
    context,
    snapshot_file,
    name_prefix="",
    volume_paths=["volumes"],
    graph_paths=["points"],
    graph_node_attrs=None,
    graph_edge_attrs=None,
    # mst=["embedding", "fg_maxima"],
    mst=None,
    roi=None,
):
    f = zarr.open(str(snapshot_file.absolute()), "r")
    with f as dataset:
        volumes = []
        for volume in volume_paths:
            volumes += get_volumes(dataset, volume)

        v = None
        for volume in volumes:
            v = daisy.open_ds(str(snapshot_file.absolute()), f"{volume}")
            if roi is not None:
                v = v.intersect(roi)
            if v.dtype == np.int64:
                v.materialize()
                v.data = v.data.astype(np.uint64)
            add_layer(context, v, f"{name_prefix}_{volume}", visible=False)
