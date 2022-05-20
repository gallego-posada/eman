# Equivariant Mesh Attention Networks

This repository contains the code to reproduce the experiments of **Equivariant Mesh Attention Networks**

## Running experiments

The instructions provided below assume that the `python` command is triggered from `./eman`:

### FAUST experiments
```
python experiments/faust_direct.py --model RelTanEMAN --seed 1 --epochs 1 -equiv_bias
```

### TOSCA experiments
```
python experiments/tosca_direct.py --model RelTanEMAN --seed 1 --epochs 1 -equiv_bias -null_isolated
```

## Installation instructions

Follow the commands below to create a new conda environment and install all dependencies:
```
conda create --name eman python=3.7
conda activate eman

# GPU installation
# conda install pytorch=1.11 cudatoolkit=11.3 -c pytorch

# CPU installation
# conda install pytorch=1.11 cpuonly -c pytorch

conda install pyg=2.0.3 -c pyg
pip install wandb pytorch-ignite openmesh opt_einsum trimesh
```

## Project structure
```
eman
│   README.md
│   LICENSE    
│
└───data
│   │   FAUST/raw/MPI-FAUST.zip  # Download from http://faust.is.tue.mpg.de/
│   │   TOSCA               # Automatically downloaded on first experiment
|
└───eman                    # Implementation of Equivariant Mesh Attention Networks
│   └───nn
│   └───tests
│   └───transform
│   └───utils
|
└───experiments
|   |   faust_direct.py 
|   |   tosca_direct.py 
|   |   paths.json          # Specify dataset locations (default: "./eman/data") 
|   |   ...
|
└───gem_cnn                 # Implementation of Gauge Equivariant CNNs
│   └───nn
│   └───tests
│   └───transform
│   └───utils
│   
└───spiralnet               # Implementation of SpiralNet++
|   |   spiralconv.py
│   └───spiralnet.utils
```

## Acknowledgements

* The code for [Gauge Equivariant Mesh CNNs](https://openreview.net/forum?id=Jnspzp-oIZE) is taken from [the official GEM-CNN implementation](https://github.com/qualcomm-ai-research/gauge-equivariant-mesh-cnn).
* The code for [SpiralNet++](SpiralNet++) comparison is taken from [the official SpiralNet++ implementation](https://github.com/sw-gong/spiralnet_plus).


