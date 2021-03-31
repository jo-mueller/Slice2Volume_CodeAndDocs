# Slice2Volume Code and Docs

This repository hosts all protcols and code that was used to process image data in the Slice2Volume [data repository](https://rodare.hzdr.de/record/904). The Slice2Volume functionality that was used to register histological slices to 3D volumes is maintained in a [separate repository](https://github.com/jo-mueller/Slice2Volume).

The following aspects are covered by this repository:

1. Histology: Staining protocols and scripts for bundle registration and image sorting (see subdirectory)
2. MRI: Scripts for denoising with BM3D, longitudinal registration and image type conversion.
3. QA: Scripts to auto-generate a LaTeX document with a compilation of all available image data and processing steps for each sample animal.
4. Simulation: Scripts for Monte-Carlo setup and subsequent zeropadding of raw simulation output to match reference CBCT dimensions.
