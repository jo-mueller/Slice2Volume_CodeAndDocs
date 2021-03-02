# Quality assurance

The script in this directory ([Generate_tex.py](https://github.com/jo-mueller/Slice2Volume_Codebase/blob/main/QA/Generate_tex.py)) iterates over all sample data on root level and automatically creates and compiles LaTeX documents for comprehensive review of included image data. The header file for the generated TeX documents is found at [QA_protocol.tex](https://github.com/jo-mueller/Slice2Volume_Codebase/blob/main/QA/tex/QA_protocol.tex). In particular, the generated pdf document includes:

* CBCT: Overview of axial/sagittal/coronal image slice.

* Simulation: Dose and LET (linear energy transfer) in keV/µm overlaid with CBCT in axial/sagittal/coronal slice. The view is centered on the centered beam axis. An additional view of the summed dose profile (*Bragg Peak*), overlaid with an axial CBCT view, is provided to estimate the beam's intended stopping location in the proximal half of the brain.

* Atlas: Axial/Sagittal/Coronal overlay of Atlas and CBCT at the beam's central location. 

* MRI: This protocol section contains:
  * Axial/sagittal/coronal views of MRI before and after BM3D denoising, separately for T1/T2-weighted sequences.
  * Gallery-overview of T1/T2-weighted images at all timepoints after longitudinal registration (axial/sagittal/coronal view).
  * Gallery-overview of T1/T2-weighted images at all timepoints with CBCT overlay after warping to the CBCT frame of reference (axial/sagittal/coronal view).
  
* Histology: This protocol section contains:
  * General overview of all detected stainings for this animal.
  * Bundle registrations: Views of the used mask images (binary or non-binary) for registration. For each step, we previde the moving image, the target image, the transformed moving image and an overlay of the target image and the transformed moving image. The script calculates mutual information and the Jaccard-coefficient to quantify the registration.
  * Slice2Volume: Panel of subsequent histological sections with DAPI staining that were transformed into the coordinate frame of the CBCT. Furthermore, an axial/sagittal/coronal view of the transformed histological stainings overlaid with the CBCT is shown.
  * Histology close-up: A whole-section and close-up detail view of all histological stainings for a selected bundle of adjacent tissue sections. The section is chosen so that it coincides with the coronal plane that is shown in the Slice2Volume overlay panel.
