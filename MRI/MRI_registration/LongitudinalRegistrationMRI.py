# -*- coding: utf-8 -*-
"""
Script to perform longitudinal registration of MRI images to day zero-image.

@author: muellejoh
"""

import os
from os.path import join
import subprocess
    
def findSequenceFile(directory, identifier):
    for root, subdirs, fnames in os.walk(directory):
        for f in fnames:
            
            # Test whether all identifers are met
            matched = True
            for ID in identifier:
                if not ID in f:
                    matched = False
                
            if matched and f.endswith('.tif'):
                return (os.path.join(root, f))
    return 0

# Params for elastix
elastix_exe = join(os.getcwd(), "..", "..", "elastix-5.0.1-win64", "elastix.exe")
transformix_exe = join(os.getcwd(), "..", "..", "elastix-5.0.1-win64", "transformix.exe")
elastix_params =  join(os.getcwd(), "elastix_parameters.txt")

base = "E:\\Promotion\\Projects\\2020_Slice2Volume\\Data\\"
scans = ["T1_GRE", "T2_FSE"]


# # Clean up the directory with denoised images first
# for root, subdirs, fnames in os.walk(base):
#     for subdir in subdirs:
        
#         # Look only at denoised directories
#         if "denoised" in subdir:
#             images = os.listdir(os.path.join(root, subdir))
#             for image in images:
                
#                 # Delete non-tifs
#                 src = os.path.join(root, subdir, image)
#                 if not image.endswith('tif'):
#                     print(src)
#                     os.remove(src)
#                 else:
#                     if 'T1' in image:
#                         dst = os.path.join(root, subdir, 'T1_GRE_SP_3D_iso_11min_T1_GRE_SP_3D_iso_11min_BM3D_15xsigma.tif')
#                     elif 'T2' in image:
#                         dst = os.path.join(root, subdir, 'T2_FSE_3D_Centric_Partial_iso_small_6min_BM3D_15xsigma.tif')
#                     os.rename(src, dst)
                
            



# Iterate over all mice
for mouse in os.listdir(base):
    print("Looking at mouse {:s}".format(mouse))
    
    if mouse != 'P2A_B6_M1':
        continue

    # iterate over all scan types
    for scan_type in scans:
        root = join(base, mouse, "MRI")
        
        # Make propper list of timepoints
        timepoints = [x for x in os.listdir(root) if "Week" in x]
        reference_time = timepoints[0]
        
        # Convert reference image once
        for tp in timepoints:
            
            # Identify first image.
            # This is registered to itself to obtain an image series of
            # similiar pixel types and greyvalues
            if tp == reference_time:
                img_fixed = findSequenceFile(join(root, tp), [scan_type, "BM3D_15xsigma"])
            
            # Make sure to continue only when reference image has been found
            if img_fixed == 0: continue
        
            # iterate over all present data collections for this timepoint
            for dfile in os.listdir(join(root, tp)):
                
                # check if we're looking at the already registered data
                if "_registered" in dfile:
                    continue
                
                # do not look at raw data
                if not "_denoised" in dfile:
                    continue
                
                # find moving image
                img_moving = findSequenceFile(join(root, tp, dfile), scan_type)
                if img_moving == 0:
                    continue
                
                # create outdir
                outdir = join(root, tp, dfile.replace("_denoised", "_registered"))
                if not os.path.isdir(outdir):
                    os.mkdir(outdir)
                    print("registering sequence {:s} timepoint: {:s} with {:s}".format(
                        scan_type, tp, reference_time), end = "")
                else:
                    print("Re-registering sequence {:s} timepoint: {:s} with {:s}".format(
                        scan_type, tp, reference_time), end = "")
                    # continue
                    
                # run Elastix
                subprocess.run([elastix_exe,
                                 "-f", img_fixed,
                                 "-m", img_moving,
                                 "-p", elastix_params,
                                 "-out", outdir], shell=True)
                
                # rename output
                trg = join(outdir,os.path.basename(img_moving).split(".")[0] + ".tif")
                if os.path.exists(trg):
                    os.remove(trg)
                os.rename(join(outdir, "result.0.tif"), trg)
                
                # rename transform parameters
                trg = join(outdir, os.path.basename(img_moving).split(".")[0] + "_trafo.txt")
                if os.path.exists(trg):
                    os.remove(trg)
                os.rename(join(outdir, "TransformParameters.0.txt"), trg)
                print("...Done")
            
    


