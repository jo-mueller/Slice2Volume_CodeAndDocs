# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 12:58:54 2021

Wrapper for the dcm2nii.exe that should be stored in the same
directory as this script.

@author: johan
"""

import os
import subprocess as sbp
import tqdm

root = 'E:\\Promotion\\Projects\\2020_Slice2Volume\\Data'
dcm2nii_exe = os.path.join(os.getcwd(), "dcm2niix.exe")

mice = [x for x in os.listdir(root) if not "zip" in x and not x.endswith('txt')]


def findnii(directory):
    
    outname = ''
    for f in os.listdir(directory):
        if f.endswith('nii'):
            outname = f
            break
    
    nii_name = outname
    json_name = outname.replace('nii', 'json')
    
    return nii_name, json_name
            

for mouse in mice:
    
    MRI_path = os.path.join(root, mouse, "MRI")
    
    timepoints = os.listdir(MRI_path)
    
    for timepoint in tqdm.tqdm(timepoints):
        timepoint_path = os.path.join(MRI_path, timepoint)
        
        if ".stfolder" in timepoint_path:
            continue

        # Find directories with raw data        
        raw_img_path = os.path.join(timepoint_path, os.listdir(timepoint_path)[0])
        
        
        if not os.path.exists(raw_img_path + "_denoised"):
            os.makedirs(raw_img_path + "_denoised")    
        denoised_img_path = os.path.join(timepoint_path,
                                         [x for x in os.listdir(timepoint_path) if "denoised" in x][0]
                                         )
        sequences = next(os.walk(raw_img_path))[1]
        
        # convert to NII
        for sequence in sequences:
            input_path = os.path.join(raw_img_path, sequence)
            
            sbp.run([dcm2nii_exe, input_path], capture_output=False)
            
            nii, json = findnii(input_path)
            
            if os.path.exists(os.path.join(denoised_img_path, nii)):
                os.remove(os.path.join(denoised_img_path, nii))
            if os.path.exists(os.path.join(denoised_img_path, json)):
                os.remove(os.path.join(denoised_img_path, json))
            
            os.rename(os.path.join(input_path, nii), 
                      os.path.join(denoised_img_path, nii))
            os.rename(os.path.join(input_path, json), 
                      os.path.join(denoised_img_path, json))
                    
            
