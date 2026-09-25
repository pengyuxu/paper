#!/usr/bin/env python3

import sys
import os
import trimesh


def main():
    csv_path = sys.argv[1]
    data_dir = sys.argv[2]

    print("id,volume")
    with open(csv_path, "r") as f:
        _ = f.readline()
        for line in f:
            if not line.strip():
                continue
            line = line.strip().split(",")
            protein_id = line[0].split(".")[0]
            mesh = trimesh.load(os.path.join(data_dir, protein_id + "_shape.obj"))
            volume = mesh.volume
            print(f"{protein_id},{volume:.4f}")


if __name__ == "__main__":
    main()
