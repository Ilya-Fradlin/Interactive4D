#!/bin/bash
export OMP_NUM_THREADS=12  # speeds up MinkowskiEngine

# Possible flags:
# --debug - Enables debug mode: "dryrun" logging, larger voxel size, and enables additional debug options...
# --num_sweeps 4 - Modify the number of scans to superimpose
# --voxel_size 0.1 - Modify the voxel size

python main.py