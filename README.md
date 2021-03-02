# Slice2Volume Codebase

This repository hosts all code that was used to process image data in the Slice2Volume [data repository](https://rodare.hzdr.de/record/801). The Slice2Volume functionality that was used to register histological slices to 3D volumes is maintained in a [separate repository](https://github.com/jo-mueller/Slice2Volume).

The following functionalities are covered by this repository:

1. Histology: Scripts for bundle registration and image sorting (see subdirectory)
2. MRI: Scripts for denoising with BM3D, longitudinal registration and image type conversion.
3. QA: Scripts to auto-generate a LaTeX document with a compilation of all available image data and processing steps for each sample animal.
4. Simulation: Scripts for zeropadding of raw simulation output to match reference CBCT dimensions.
