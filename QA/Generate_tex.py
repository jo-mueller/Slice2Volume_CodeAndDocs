"""
Created on Sun Jan 24 13:54:42 2021

This script browses S2V data directories and creates .tex files
that contain an overview over the contents of each directory.
These overview protocols can then be used for QA.
"""
import os
import numpy as np
import tifffile as tf
from tqdm import tqdm
import scipy.ndimage as ndimage
import subprocess

from qa_utils import read_CT,\
    create_Title,\
    create_CBCT,\
    create_Atlas,\
    create_Histology,\
    create_DoseLET,\
    create_MRI,\
    create_Info,\
    copy_and_overwrite

if __name__ == '__main__':

    # necessary directories
    root = "E:\\Promotion\\Projects\\2020_Slice2Volume\\Data\\"
    tex_dir = os.path.join(os.getcwd(), "tex")

    cutplanes = [87, 160, 100]

    # Get list of mice
    mice = [x for x in os.listdir(root)  if not "." in x]

    for mouse in mice:

        # Let's only do this mouse
        if mouse != "P2A_B6_M1":
            continue

        print(" ", flush=True)
        print(mouse, flush=True)

        # Make QA directory if it doesn't exist
        QA_dir = os.path.join(root, mouse, "QA")
        if not os.path.isdir(QA_dir):
            os.makedirs(QA_dir)

        # # Move tex header file to the created QA dir
        local_tex_dir = os.path.join(QA_dir, "tex")
        CBCT_dir = os.path.join(root, mouse, "CT")
        Atlas_dir = os.path.join(root, mouse, "Atlas")
        MRI_dir = os.path.join(root, mouse, "MRI")
        Dose_dir = os.path.join(root, mouse, "Simulation")
        Histo_dir = os.path.join(root, mouse, "Histology")
        info_dir = os.path.join(os.getcwd(), 'resources')

        if not os.path.exists(CBCT_dir):
            continue

        # read CBCT
        CBCT = read_CT(CBCT_dir)

        # Find a good cutplane and boundary volume
        if os.path.exists(Dose_dir):
            # If dose files exist, take beam center for cutplane
            Dose = tf.imread(os.path.join(Dose_dir, 'Dose.tif'))
            Dose = np.einsum('ijk->jki', Dose)

            # Project dose
            projected_sag_dose = np.sum(Dose, axis=1)
            z = np.argmax(np.sum(projected_sag_dose, axis=0))
            y = np.argmax(np.sum(projected_sag_dose, axis=1))
            x = int(CBCT.shape[1]/2)
            cutplanes = [y, x, z]

        else:
            Atlas = tf.imread(os.path.join(Atlas_dir, 'DSURQE_40micron_labels_transformed_resliced.tif'))
            Atlas[Atlas>0] = 1
            Atlas = Atlas.transpose((0,2,1))

            cutplanes = np.asarray(ndimage.center_of_mass(Atlas), dtype=int)

        copy_and_overwrite(tex_dir, local_tex_dir)
        create_Title(mouse, local_tex_dir)
        create_Info(mouse, info_dir, local_tex_dir)
        create_CBCT(CBCT, local_tex_dir, cutplanes=cutplanes)
        boundaries = create_Atlas(Atlas_dir, CBCT, local_tex_dir, cutplanes=cutplanes)
        create_DoseLET(Dose_dir, CBCT, local_tex_dir, cutplanes=cutplanes, mouse=mouse)
        create_MRI(MRI_dir, CBCT, local_tex_dir, cutplanes, boundaries)
        create_Histology(Histo_dir, CBCT, local_tex_dir, boundaries=boundaries)

        # Compile tex file
        os.chdir(local_tex_dir)
        for i in tqdm(range(0,2), desc = 'Compiling document'):
            subprocess.call(['pdflatex',
                             '-jobname=' + mouse + '_QA_protocol',
                             '-synctex=1',
                             '-interaction=nonstopmode',
                             '-output-directory=' + local_tex_dir,
                             'QA_protocol.tex'],
                            shell=True)
