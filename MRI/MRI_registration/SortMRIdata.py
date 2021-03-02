# -*- coding: utf-8 -*-
"""

Browses a bunch of MRI raw data folders and sorts the dicom images
in subdirectories named after the respective used sequence.

Created on Fri Oct 16 16:44:29 2020

@author: muellejoh
"""

import os
import pydicom as dcm
import shutil
import SimpleITK as sitk

def slugify(string):
    string = string.replace(" ", "_")
    string = string.replace("-", "_")
    string = string.replace("(", "")
    string = string.replace(")", "")
    return string

def DCMconvert(input_dir, output_file):

    print("Reading Dicom directory:", input_dir)
    reader = sitk.ImageSeriesReader()
    
    dicom_names = reader.GetGDCMSeriesFileNames(input_dir)
    reader.SetFileNames(dicom_names)
    
    image = reader.Execute()
    
    size = image.GetSize()
    print("Image size:", size[0], size[1], size[2])
    print("Writing image:", output_file)    
    sitk.WriteImage(image, output_file)

root = "E:/2020_Slice2Volume/Data/"

mice = os.listdir(root)

for mouse in mice:
    MRI_dir = os.path.join(root, mouse, "MRI")
    for base, subdirs, files in os.walk(MRI_dir):
        
        # Skip empty subdirs
        if len(files) == 0:
            continue
        
        print("Sorting {:s}".format(base))
        n_sorted = 0
        n_skipped = 0
        for f in files:
            
            # Try to read dicom meta data. Skip if this fails
            try:
                meta = dcm.read_file(os.path.join(base, f))
                SeqName = slugify(meta.SeriesDescription)
            except Exception:
                continue
            
            # Check if this file has already been sorted. This is the case
            # if the parent directory is named after the sequence
            # Additionally check if the directory has been converted to a
            # single file yet
            parent_dir = os.path.normpath(base).split("\\")[-1]
            dcm_file = os.path.join(os.path.dirname(base), SeqName + ".dcm")
            if parent_dir == SeqName:
                n_skipped += 1
                if not os.path.exists(dcm_file):
                    DCMconvert(base, dcm_file)
                break
                
            
            # Check if directory for this sequence already exists
            seqpath = os.path.join(base, SeqName)
            if not os.path.exists(seqpath):
                os.mkdir(seqpath)
            
            src = os.path.join(base, f)
            
            # add dicom ending to file if not present
            if not f.endswith(".dcm"):
                f = f + ".dcm"
            dst = os.path.join(seqpath, f)
            
            # Copy to new destination
            shutil.move(src, dst)
            n_sorted += 1
        N = n_sorted + n_skipped
        if N > 0:
            print("\t Sorted {:d} out of {:d} files".format(n_sorted, N))