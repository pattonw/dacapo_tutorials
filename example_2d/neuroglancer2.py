import daisy
import neuroglancer
import numpy as np
import zarr
import h5py

import logging

logger = logging.getLogger(__name__)


def add_layer(context, array, name, voxel_size, array_offset, visible=True, **kwargs):
    array_dims = len(array.shape)
    spatial_dims = len(voxel_size)
    assert spatial_dims == 2
    channel_dims = array_dims - spatial_dims
    spatial_attrs = {
        2: {"names": ["y", "x"], "units": ["nm", "nm"], "scales": list(voxel_size)},
        3: {
            "names": ["z", "y", "x"],
            "units": ["nm", "nm", "nm"],
            "scales": list(voxel_size),
        },
    }
    channel_attrs = {
        0: {
            "names": [],
            "units": [],
            "scales": [],
        },
        1: {
            "names": ["c^"],
            "units": [""],
            "scales": [1],
        },
        2: {
            "names": ["c^", "b^"],
            "units": ["", ""],
            "scales": [1, 1],
        },
    }
    attrs = {
        array_dims: {
            k: v_channel + v_spatial
            for k, v_channel, v_spatial in zip(
                channel_attrs[channel_dims].keys(),
                channel_attrs[channel_dims].values(),
                spatial_attrs[spatial_dims].values(),
            )
        }
    }
    dimensions = neuroglancer.CoordinateSpace(**attrs[array_dims])
    offset = np.array((0,) * (channel_dims) + array_offset)
    offset = offset // attrs[array_dims]["scales"]

    if channel_dims > 0 and array.dtype == np.uint8:
        array = array.astype(np.float32) / 255
    print(name, array.shape, array.dtype)

    channels = ",".join(
        [
            f"toNormalized(getDataValue({i}))" if i < array.shape[0] else "0"
            for i in range(3)
        ]
    )
    shader_1_channel = (
        """
void main() {
  emitRGB(vec3(%s));
}
"""
        % channels
    )
    shader_no_channels = """
void main () {
  emitGrayscale(toNormalized(getDataValue()));
}"""

    shaders = {0: shader_no_channels, 1: shader_1_channel}

    layer = neuroglancer.LocalVolume(
        data=array, dimensions=dimensions, voxel_offset=tuple(offset)
    )

    if array.dtype == np.dtype(np.uint64):
        context.layers.append(name=name, layer=layer, visible=visible)
    else:
        context.layers.append(
            name=name,
            layer=layer,
            visible=visible,
            shader=shaders[channel_dims],
            **kwargs,
        )


def get_volumes(h5_file, path=None):
    datasets = []
    try:
        if path is None:
            for key in h5_file.keys():
                datasets += get_volumes(h5_file, f"{key}")
            return datasets
        else:
            for key in h5_file.get(path, {}).keys():
                datasets += get_volumes(h5_file, f"{path}/{key}")
            return datasets
    except AttributeError:
        return [path]


def add_container(
    context,
    snapshot_file,
    name_prefix="",
    volume_paths=[None],
    graph_paths=[None],
    graph_node_attrs=None,
    graph_edge_attrs=None,
    # mst=["embedding", "fg_maxima"],
    mst=None,
    roi=None,
    modify=None,
    dims=3,
):
    if snapshot_file.name.endswith(".zarr") or snapshot_file.name.endswith(".n5"):
        f = zarr.open(str(snapshot_file.absolute()), "r")
    elif snapshot_file.name.endswith(".h5") or snapshot_file.name.endswith(".hdf"):
        f = h5py.File(str(snapshot_file.absolute()), "r")
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
            if v.dtype == np.dtype(bool):
                v.materialize()
                v.data = v.data.astype(np.float32)

            v.materialize()
            if modify is not None:
                data = modify(v.data, volume)
            else:
                data = v.data

            add_layer(
                context,
                data,
                f"{name_prefix}_{volume}",
                visible=False,
                voxel_size=v.voxel_size[-dims:],
                array_offset=v.roi.get_offset()[-dims:],
            )


def add_dacapo_snapshot(
    context,
    snapshot_file,
    name_prefix="",
    volume_paths=[None],
    graph_paths=[None],
    graph_node_attrs=None,
    graph_edge_attrs=None,
    # mst=["embedding", "fg_maxima"],
    mst=None,
    roi=None,
):
    def modify(v, name):
        if name == "prediction":
            v = reshape_batch_channel(v, 1, 3)
        elif name == "raw":
            v = reshape_batch_channel(v, 0, 2)
        if name == "target":
            v = reshape_batch_channel(v, 1, 3)
        return v

    add_container(
        context,
        snapshot_file,
        name_prefix,
        volume_paths,
        graph_paths,
        graph_node_attrs,
        graph_edge_attrs,
        mst,
        roi,
        modify,
    )


def reshape_batch_channel(array, batch_dim=0, concat_dim=0):
    # Given shape (a0, a1, ..., am) and batch dim k:
    # First remove the dim ak: new_shape = (a0, a1, ..., ak-1, ak+1, ..., am)
    # Next replace concat_dim with -1

    if batch_dim is not None:
        array = np.concatenate(np.rollaxis(array, batch_dim), concat_dim)
    return array