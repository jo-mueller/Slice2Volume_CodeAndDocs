# Histology processing

This directory stores all staining protocols and scripts that were used for histological processing and bundle registration of adjacent histology-slices. 

### Protocols:
* HE-stain.pdf: Materials and procedures for H&E stain
* IF staining protocol.pdf: Materials and procedures for IF stain, can be adopted for various markers.

### Scripts:
* Bundle_registration.ijm: iterates over all raw images (must be stored in separate directories) and registers them with [Elastix](https://elastix.lumc.nl/) [1, 2]. The script creates downsampled versions of the raw image files. If moving and fixed image are immunofluorescent images, the DAPI channel (present in both) is used to create cell-density masks that are then used for registration. If moving and image are immunofluorescent and immunohistochemical images, respectively, the script creates binary masks for the registration. The registration is then carried out at low resolution and the resulting transformation parameters are adjusted to match the high-resolution images.
* SortHistoSlices.py: is a helper Python-script that takes raw czi (Zeiss Microscopy) images and sorts these images into separate (bundle-wise) directories. For example see script header.
* Elastix_parameters.txt: Elastix-readable parameter file for the similarity-based registration of adjacent histological samples based on downsampled images.



### References
[1] S. Klein, M. Staring, K. Murphy, M.A. Viergever, J.P.W. Pluim, "elastix: a toolbox for intensity based medical image registration," IEEE Transactions on Medical Imaging, vol. 29, no. 1, pp. 196 - 205, January 2010

[2] D.P. Shamonin, E.E. Bron, B.P.F. Lelieveldt, M. Smits, S. Klein and M. Staring, "Fast Parallel Image Registration on CPU and GPU for Diagnostic Classification of Alzheimer's Disease", Frontiers in Neuroinformatics, vol. 7, no. 50, pp. 1-15, January 2014
