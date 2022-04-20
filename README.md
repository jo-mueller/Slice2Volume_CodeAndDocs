# Slice2Volume Code and Docs

This repository hosts all protcols and code that was used to process image data in the Slice2Volume [data repository](https://rodare.hzdr.de/record/915). The Slice2Volume functionality that was used to register histological slices to 3D volumes is maintained in a [separate repository](https://github.com/jo-mueller/Slice2Volume).

The following aspects are covered by this repository:

1. Histology: Staining protocols and scripts for bundle registration and image sorting (see subdirectory)
2. MRI: Scripts for denoising with BM3D, longitudinal registration and image type conversion.
3. QA: Scripts to auto-generate a LaTeX document with a compilation of all available image data and processing steps for each sample animal.
4. Simulation: Scripts for Monte-Carlo setup and subsequent zeropadding of raw simulation output to match reference CBCT dimensions.

## Installation
To install or use the Python-related functionality, install Anaconda as described [here](https://biapol.github.io/blog/johannes_mueller/anaconda_getting_started/) and create a new environment with:

```
conda create -n slice2volume Python=3.9
conda activate slice2volume
```

Then clone the repository using git clone (`git clone https://github.com/jo-mueller/Slice2Volume_CodeAndDocs.git`) or download the repository with any other method of your choice. Use the command line to navigate into the downloaded directory and install the necessary packages:

```
cd path_to_repository/Slice2Volume_CodeAndDocs
<<<<<<< Updated upstream
conda install - c conda-forge napari jupyterlab
=======

conda install -c conda-forge napari

>>>>>>> Stashed changes
pip install -r requirements.txt
```

To start jupyterlab, type `jupyter-lab` in the command line and navigate to the location of nnotebooks you wish to execute.
