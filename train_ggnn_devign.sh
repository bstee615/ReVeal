#!/bin/bash
sbatch batch.sh bash -x ggnn.sh -i "$PWD/data_refactored/chrome_debian" -o "$PWD/out_refactored/" -n chrome_debian
sbatch batch.sh bash -x devign.sh -i "$PWD/data_refactored/chrome_debian" -o "$PWD/out_refactored/" -n chrome_debian
