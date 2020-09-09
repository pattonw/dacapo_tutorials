import neuroglancer
import sys
from pathlib import Path

from neuroglancer2 import add_snapshot


neuroglancer.set_server_bind_address("0.0.0.0")



if __name__ == "__main__":
    args = sys.argv[1:]
    snapshot_file = args[0]
    if len(args) >= 2:
        graphs = args[1].split(",")
    else:
        graphs = ["points"]

    if len(args) >= 3:
        volumes = args[2].split(",")
    else:
        volumes = ["volumes"]

    voxel_size = [1000, 300, 300]

    dimensions = neuroglancer.CoordinateSpace(
        names=["z", "y", "x"], units="nm", scales=voxel_size
    )

    snapshot_files = snapshot_file.split(",")

    viewer = neuroglancer.Viewer()
    viewer.dimensions = dimensions

    print(graphs)
    print(volumes)

    with viewer.txn() as s:
        for snapshot_file in snapshot_files:
            print(snapshot_file)
            add_snapshot(s, Path(snapshot_file), graph_paths=graphs, volume_paths=volumes)

    print(viewer)
    input("Hit ENTER to quit!")
