# -*- coding: utf-8 -*-
"""
This script reads the raw results of simulated Dose and LET from the directories.
The simulated array is padded with zeros to match the dimensions of the respective 
CBCT image. The dose is rescaled to relative units [0...1], the LET is given
in keV/m E10-6.
The simulation was split in 10 independent runs that are condensed into a single
output file in this script.

Created on Mon Feb  1 17:02:57 2021

@author: johannes.mueller@hzdr.de
"""

import os
import numpy as np
import pydicom as dcm
import tqdm
import random
import tifffile as tf

def read_CT(directory):
    """
    Read dcm files from a directory and fuse to array
    """
    
    # get CBCT data
    slices = os.listdir(directory)
    
    meta = dcm.read_file(os.path.join(directory, slices[0]))
    Array = np.zeros((meta.Rows, meta.Columns, len(slices)))
    
    for i, slc in tqdm.tqdm(enumerate(slices)):
        meta = dcm.read_file(os.path.join(directory, slc))
        Array[:, :, i] = meta.pixel_array
        
    return Array

"""
BL6:

Maus   CTAuswahlMitte   CTAuswahlMitte-45  CTAuswahlMitte+45 
---- ------------------ ------------------ ------------------
   1       98                  53                   143
   2       89                  44                   134
   6       97                  52                   142
  10       97                  52                   142

C3H:

Maus   CTAuswahlMitte   CTAuswahlMitte-45  CTAuswahlMitte+45 
---- ------------------ ------------------ ------------------
  1         122               77                  167
  3         116               71                  161
  5         118               73                  163
  8         130               85                  175
 10         128               83                  173 


"""

first = 83  # first simulated axial slice
N_slices = 91   # number of simulated axial slices

if __name__ == '__main__':
    
    root = r"E:\Promotion\Projects\2020_Slice2Volume\Data\P2A_C3H_M10"
    CBCT = read_CT(os.path.join(root, "CT"))
    
    Dose = np.zeros_like(CBCT)
    LET = np.zeros_like(CBCT)
    
    # Dose first
    Doses = []
    LETs = []
    for base, subdirs, fnames in os.walk(os.path.join(root, "Simulation")):
        for fname in fnames:
            if fname.endswith("dcm") and "TotalDose" in fname:
                Doses.append(os.path.join(base, fname))
            elif fname.endswith("dcm") and "ProtonLET" in fname:
                LETs.append(os.path.join(base, fname))
    
    # Pick 10 out of the >10 statistically independent experiment runs
    Doses = random.sample(Doses, 10)
    LETs = random.sample(LETs, 10)
    
    for i in range(len(Doses)):
        meta_LET = dcm.read_file(LETs[i])
        meta_dose = dcm.read_file(Doses[i])
        
        meta_LET = np.einsum('ijk -> jki', meta_LET.pixel_array) * float(meta_LET.DoseGridScaling)
        meta_dose = np.einsum('ijk -> jki', meta_dose.pixel_array) * float(meta_dose.DoseGridScaling)
        
        Dose[:, :, first:first+N_slices] += meta_dose
        LET[:, :, first:first+N_slices] += meta_LET
    
    # Average LET
    LET = LET/len(LET)
    
    Dose = (Dose - np.min(Dose))/(Dose.max() - Dose.min())
    
    tf.imwrite(os.path.join(root, "Simulation", "Dose.tif"), data=np.einsum('jki -> ijk', Dose))
    tf.imwrite(os.path.join(root, "Simulation", "LET.tif"),  data=np.einsum('jki -> ijk', LET))
        
    