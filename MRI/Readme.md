# MRI image processing

This directory contains script that were used to process MRI images from longitudinal follow-up. These steps include:
1. Denoising
2. Longitudinal registration to timepoint zero image

## Denoising
The functionality used for the denoising can be accessed [here](https://www.cs.tut.fi/~foi/GCF-BM3D/) [1]. For the denoising, the images were first converted from the raw dicom format to niftii with [dcm2niix](https://www.nitrc.org/plugins/mwiki/index.php/dcm2nii:MainPage) with the script MRI_conversion/Convert2NII.py. The converted images were then denoised with the linked BM3D algorithm. The output images were converted into tif-format with the script MRI_conversion/ConvertNII2TIF.ijm in order to be consistent with other data in the repository.

## Longitudinal registration
We used the command-line registration [Elastix](https://elastix.lumc.nl/)[2,3] for affine registration of MRI images. All denoised T1-weighted images from timepoint t (MRI<sub>t</sub>) were registered with the reference image (MRI<sub>0</sub>). The elastix parameter file is saved at [elastix_parameters.txt](https://github.com/jo-mueller/Slice2Volume_Codebase/blob/main/MRI/MRI_registration/elastix_parameters.txt). The process is called through the Python script [LongitudinalRegistrationMRI.py](https://github.com/jo-mueller/Slice2Volume_Codebase/blob/main/MRI/MRI_registration/LongitudinalRegistrationMRI.py).

Helper scripts:
* As raw data, dicom slices from all sequences of the same study are thrown into the same directory, [SortMRIdata.py](https://github.com/jo-mueller/Slice2Volume_Codebase/blob/main/MRI/MRI_misc/SortMRIdata.py) groups these into separate folders.
* To get an impression over the longitudinal registration, [View4DMRI.ijm](https://github.com/jo-mueller/Slice2Volume_Codebase/blob/main/MRI/MRI_registration/View4DMRI.ijm) creates a hyperstack of all registered T1/T2-weighted images along the time-axis.



### References:
[1] Image denoising by sparse 3-D transform-domain collaborative filtering. Dabov K, Foi A, Katkovnik V, Egiazarian K IEEE Trans Image Process. 2007 Aug; 16(8):2080-95.

[2] S. Klein, M. Staring, K. Murphy, M.A. Viergever, J.P.W. Pluim, "elastix: a toolbox for intensity based medical image registration," IEEE Transactions on Medical Imaging, vol. 29, no. 1, pp. 196 - 205, January 2010

[3] D.P. Shamonin, E.E. Bron, B.P.F. Lelieveldt, M. Smits, S. Klein and M. Staring, "Fast Parallel Image Registration on CPU and GPU for Diagnostic Classification of Alzheimer's Disease", Frontiers in Neuroinformatics, vol. 7, no. 50, pp. 1-15, January 2014
