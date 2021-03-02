# -*- coding: utf-8 -*-
"""

Sorts Histological images in separate directory according to the section location.
It is necessary for each file to contain the tag XXXX_Scene_Y to ensure
propper sorting.

Example - Raw images name as follows:
    MyStaining_1_0001_Scene_1.czi
    MyStaining_1_0001_Scene_2.czi
    AnotherStaining_0001_Scene_1.czi
    AnotherStaining_0001_Scene_2.czi
    
Will be grouped as:
    0001_Scene_1/
        MyStaining_1_0001_Scene_1.czi
        AnotherStaining_0001_Scene_1.czi
    0001_Scene_2/
        MyStaining_1_0001_Scene_2.czi
        AnotherStaining_0001_Scene_2.czi
        
        
Created on Wed Oct 14 11:06:17 2020
@author: johannes.mueller@hzdr.de
"""

import os
import tqdm
from itertools import chain
from shutil import copyfile
import re

def parsename(string):
    "Extracts propper dirname from histo filename"
    substrings = re.split("-|_|\.", string)
    for i in range(len(substrings)):
        if substrings[i] == "Scene":
            break
    SliceNumber = substrings[i-1]
    try:
        SceneNumber = substrings[i+1]
    except Exception.IndexError:
        print("Script failed here:")
        print(substrings)
    
    return "_".join([SliceNumber, "Scene", SceneNumber])

root = "E:/Promotion/Projects/2020_Slice2Volume/Data/"

mouselist = os.listdir(root)

for mouse in mouselist:
    print(mouse)
    
    if not mouse == "P2A_C3H_M5":
        continue
    
    if not os.path.exists(os.path.join(root, mouse, "Histo")):
        continue
    
    print("Sorting data for mouse " + mouse)
    dirlist = os.listdir(os.path.join(root, mouse, "Histo"))
    histolist = [d for d in dirlist if not "Scene" in d and not d.endswith("czi")]
    slicelist = os.listdir(os.path.join(root, mouse, "Histo", histolist[0]))
    
    # make directories
    for i in range(len(slicelist)):
        
        if not "Scene" in slicelist[i]:
            continue
        name = parsename(slicelist[i])
        trgpath = os.path.join(root, mouse, "Histo", name)
        
        if not os.path.isdir(trgpath):
            os.mkdir(trgpath)
        
    # Now that all folders are created, data to correct location
    histolist = [os.path.join(root, mouse, "Histo", x) for x in histolist]
    for base, subdirs, files in chain.from_iterable(os.walk(x) for x in histolist):
        
        # Skip empty directories
        if len(files) == 0:
            continue
        for f in tqdm.tqdm(files):
            
            # Skip this blasted file
            if f == "Thumbs.db":
                continue
            parsedname = parsename(f)
            
            # propper fileformatting
            img_type = (''.join([x[0] for x in os.path.basename(base).split('-')]) +
                        "_" + os.path.basename(base).replace("-", "_"))
            src = os.path.join(base, f)
            _f = f.replace('-', '_').split('_')
            _f = '_'.join(_f[0:4] + [img_type] + _f[4:])
            trg = os.path.join(root, mouse, "Histo", parsedname, _f)
            if os.path.exists(trg):
                continue

            copyfile(src, trg) 


    