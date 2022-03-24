import copy
import os
import shutil
import numpy as np
import tifffile as tf
import matplotlib.pyplot as plt
import pydicom as dcm
from functools import partial
from tqdm import tqdm
import scipy.ndimage as ndimage
from scipy.signal import find_peaks
from scipy.stats import ttest_ind
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
import subprocess
from skimage.filters import threshold_triangle
import pandas as pd


from sklearn.metrics import normalized_mutual_info_score
tqdm = partial(tqdm, position=0, leave=True)
c = mcolors.ColorConverter().to_rgb
eps = np.finfo(float).eps

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def get_cmap(name, image):
    """
    Takes filename of certain staining and creates the propper colormap for it
    """
    if 'DAPI' in name:
        cmap = make_colormap([c('black'), c('blue')])
        vmin = np.quantile(image, 0.5)
        vmax = np.quantile(image, 0.95)
    elif 'HE' in name:
        cmap = None
        vmin = None
        vmax = None
    elif 'OSP' in name:
        cmap = make_colormap([c('black'), c('brown')])
        vmin = np.quantile(image, 0.4)
        vmax = np.quantile(image, 0.95)
    elif 'GFAP' in name:
        cmap = make_colormap([c('black'), c('green')])
        vmin = np.quantile(image, 0.6)
        vmax = np.quantile(image, 0.98)
    elif 'Iba1' in name:
        cmap = 'YlOrBr_r'
        vmin = np.quantile(image, 0.5)
        vmax = np.quantile(image, 0.98)
    elif 'Nestin' in name:
        cmap = make_colormap([c('black'), c('cyan')])
        vmin = np.quantile(image, 0.6)
        vmax = np.quantile(image, 0.98)
    elif 'Ki67' in name:
        cmap = make_colormap([c('black'), c('orange')])
        vmin = np.quantile(image, 0.8)
        vmax = np.quantile(image, 0.999)
    elif 'NeuN' in name:
        cmap = 'PuBuGn_r'
        vmin = np.quantile(image, 0.55)
        vmax = np.quantile(image, 0.98)
    elif 'Cas3' in name:
        cmap = make_colormap([c('black'), c('purple')])
        vmin = threshold_triangle(image)
        vmax = np.quantile(image, 0.98)
    elif 'CD45' in name:
        cmap = make_colormap([c('black'), c('magenta')])
        vmin = threshold_triangle(image)
        vmax = np.quantile(image, 0.98)
    elif 'HIF1a' in name:
        cmap = make_colormap([c('black'), c('white')])
        vmin = np.quantile(image, 0.6)
        vmax = np.quantile(image, 0.98)
    elif 'MAP2' in name:
        cmap = 'inferno'
        vmin = np.quantile(image, 0.6)
        vmax = np.quantile(image, 0.98)

    return cmap, vmin, vmax

def MutualInformation2(ImgA, ImgB, bins = 50):

    hist, xedges, yedges = np.histogram2d(ImgA.flatten(), ImgB.flatten(), bins=50)

    pxy = hist/np.sum(hist)

    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)

    nzsx = px > 0
    nzsy = py > 0

    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals

    Ex = -(px[nzsx]*np.log(px[nzsx])).sum()
    Ey = -(py[nzsy]*np.log(py[nzsy])).sum()

    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum

    MI = (pxy[nzs] * np.log(pxy[nzs] / px_py[nzs])).sum()
    MI_norm = 2*MI / (Ex + Ey)

    return MI_norm


def MutualInformation(ImgA, ImgB, method='arithmetic', sample_size=50000,
                      sigma=15):
    """
    Returns mutual information between image A and Image B as raw value
    and normalized by the minimum of the entropy of Image A and Image B
    min(E(ImgA), E(ImgB))
    """

    idx = np.random.randint(0, len(ImgA), sample_size)

    ImgA_fltrd = gaussian_filter(ImgA, sigma = sigma).flatten()[idx]
    ImgB_fltrd = gaussian_filter(ImgB, sigma = sigma).flatten()[idx]

    # MI = mutual_info_score(ImgA_fltrd, ImgB_fltrd)
    MI_norm = normalized_mutual_info_score(ImgA_fltrd,
                                           ImgB_fltrd,
                                           average_method=method)
    return MI_norm

def latexify(string):
    string = string.replace("_", "\\textunderscore ")
    return string

def itemize(mylist):
    command = ""

    command += "\\begin{itemize}\n"
    for item in mylist:
        command += "\t\\item {:s}\n".format(latexify(str(item)))
    command += "\\end{itemize}\n"

    return command

def latex_figure(img_fname, subdir="", caption="", width=1.0):
    """
    returns a formatted string for a latex-readable figure that can be put
    into a given document
    """

    img_fname = os.path.basename(img_fname)
    command = ["\\begin{figure}[htb]\n",
               "\t\\centering\n",
               "\t\\includegraphics[width={:.1f}\\textwidth]{{Images/{:s}{:s}}}\n".format(width, subdir, img_fname),
               "\t\\caption{{{:s}}}\n".format(caption),
               "\\end{figure}\n"]
    return ''.join(command)

def copy_and_overwrite(from_path, to_path):
    if os.path.exists(to_path):
        shutil.rmtree(to_path)
    shutil.copytree(from_path, to_path)

def get_bounding_box(ndarray, margin=10, perc=0.2):
    "Determines coordinates for an oobject in an nd-array"


    boundaries = []

    for i, dim in enumerate(ndarray.shape):

        # Sum up image along relevant dimension
        array =np.sum(np.sum(ndarray, axis=i), axis=i%2)
        _array = np.zeros_like(array, dtype='float64')

        # find edges in summed profiles with derivative
        threshold = np.min(array) + perc * (np.max(array) - np.min(array))
        _array[array>threshold] = 1.0
        _array[array<threshold] = 0.0

        array = abs(np.diff(_array))
        boundaries.append(find_peaks(array)[0])

    # check if boundaries agree with array dimensions
    boundaries = boundaries[::-1]
    for i in range(len(boundaries)):

        boundaries[i][0] -= margin
        boundaries[i][1] += margin

        if boundaries[i][0] < 0:
            boundaries[i][0] = 0

        if boundaries[i][1] > ndarray.shape[i]:
            boundaries[i][1] = ndarray.shape[i]


    return boundaries

def read_CT(directory):

    #get CBCT data
    slices=  os.listdir(directory)

    meta = dcm.read_file(os.path.join(directory, slices[0]))
    Array = np.zeros((meta.Rows, meta.Columns, len(slices)))

    for i, slc in enumerate(tqdm(slices, desc='Reading CBCT')):
        meta = dcm.read_file(os.path.join(directory, slc))
        Array[:, :, i] = meta.pixel_array

    return Array

def multiview_overlay(Array1, Array2=None, **kwargs):

    # get kwargs
    cmap1 = kwargs.get('cmap1', 'gray')
    cmap2 = kwargs.get('cmap2', 'inferno')
    vmin1 = kwargs.get('vmin1', None)
    vmax1 = kwargs.get('vmax1', None)

    title1 = kwargs.get('Title1', "")
    title2 = kwargs.get('Title2', "")

    alpha = kwargs.get('alpha', 0.7)
    cutplanes = kwargs.get('cutplanes', None)

    show_cross = kwargs.get('show_cross', False)

    if "bds" in kwargs:
        #bds = np.asarray([(0, x) for x in Array1.shape])
        bds = kwargs["bds"]
        # x0 = bds[0,1]
        Array1 = Array1[bds[0][0] : bds[0][1],
                        bds[1][0] : bds[1][1],
                        bds[2][0] : bds[2][1]]

        if Array2 is not None:
            Array2 = Array2[bds[0][0] : bds[0][1],
                            bds[1][0] : bds[1][1],
                            bds[2][0] : bds[2][1]]

    if "ax" not in kwargs:
        fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
    else:
        ax = kwargs.get('ax', None)

    if cutplanes is None:
        if Array2 is not None:
            _Array = (Array2 - Array2.min())/(Array2.max() - Array2.min())
        else:
            _Array = (Array1 - Array1.min())/(Array1.max() - Array1.min())
        cutplanes=np.asarray(ndimage.center_of_mass(_Array)).astype(int)
        kwargs["cutplanes"] = cutplanes

    cp = cutplanes


    ax[0].imshow(Array1[cp[0], :, :], cmap=cmap1, vmin=vmin1, vmax=vmax1)
    ax[1].imshow(Array1[:, cp[1], :], cmap=cmap1, vmin=vmin1, vmax=vmax1)
    ax[2].imshow(Array1[:, :, cp[2]], cmap=cmap1, vmin=vmin1, vmax=vmax1)

    if Array2 is not None:
        vmin2 = kwargs.get('vmin2', None)
        vmax2 = kwargs.get('vmax2', None)
        ax[0].imshow(Array2[cp[0], :, :], cmap=cmap2, alpha=alpha, vmin=vmin2, vmax=vmax2)
        ax[1].imshow(Array2[:, cp[1], :], cmap=cmap2, alpha=alpha, vmin=vmin2, vmax=vmax2)
        ax[2].imshow(Array2[:, :, cp[2]], cmap=cmap2, alpha=alpha, vmin=vmin2, vmax=vmax2)

    # Show cross-sectional lines?
    if show_cross:
        ax[0].hlines(cp[1], 0, ax[0].get_xlim()[1], color='blue', linestyle='--')
        ax[0].vlines(cp[2], 0, ax[0].get_ylim()[0], color='blue', linestyle='--')

        ax[1].hlines(cp[0], 0, ax[1].get_xlim()[1], color='blue', linestyle='--')
        ax[1].vlines(cp[2], 0, ax[1].get_ylim()[0], color='blue', linestyle='--')

        ax[2].hlines(cp[0], 0, ax[2].get_xlim()[1], color='blue', linestyle='--')
        ax[2].vlines(cp[1], 0, ax[2].get_ylim()[0], color='blue', linestyle='--')

    # Print title?
    if title1 != "" and title2 != "":
        ax[0].set_title("{:s} and {:s} \ncoronal".format(title1, title2), multialignment='center')
        ax[1].set_title("{:s} and {:s} \nsagittal".format(title1, title2), multialignment='center')
        ax[2].set_title("{:s} and {:s} \naxial".format(title1, title2), multialignment='center')
        return ax[0].get_figure(), kwargs

    elif title1 != "" and title2 == "":
        ax[0].set_title("{:s} \ncoronal".format(title1), multialignment='center')
        ax[1].set_title("{:s} \nsagittal".format(title1), multialignment='center')
        ax[2].set_title("{:s} \naxial".format(title1), multialignment='center')
        return ax[0].get_figure(), kwargs

def create_Info(mouse: str, info_dir: str, dst: str):

    info = pd.read_csv(os.path.join(info_dir, 'animals.csv'), sep = ';')
    info = info[info["Animal-code"] == mouse]  # get relevant dataframe row

    info['Dose'] = info['Dose'].astype(str)

    f = os.path.join(dst, "Info.tex")
    with open(f, 'wt') as file:
        file.write("\\section{Animal information}\n")
        for col in info.columns:
            file.write(f"{latexify(col)}: {latexify(info[col].loc[0])}\\\\\n")
    file.close()

def create_Title(sample, dst):
    """
    Generate Title.tex file for a given sample at location

    """
    sample = latexify(sample)

    f = os.path.join(dst, "Title.tex")
    with open(f, 'wt') as file:
        file.write("\\title{{QA protocol: {:s}}}".format(sample))

    file.close()

def create_CBCT(CBCT, dst, cutplanes):
    """
    Creates latex--readable documentation/overview for CBCT data
    """
    fig,_  = multiview_overlay(CBCT, cutplanes=cutplanes,
                               cmap1='gray', Title1="CBCT",
                               vmin1=-500, vmax1=200)

    fig_fname = os.path.join(dst, 'Images', 'CBCT.png')
    fig.savefig(fig_fname, dpi=150)
    plt.close(fig)
    caption="Multi-view cutplanes of CBCT serving as anatomical reference\n"

    # Make CBCT.tex
    f = os.path.join(dst, "CBCT.tex")
    with open(f, 'wt') as file:
        file.write("\\section{CBCT image data}\n")
        file.write(latex_figure(fig_fname, caption=caption))
        file.write("\\FloatBarrier")

    file.close()

def create_Atlas(Atlas_dir, CBCT, dst, cutplanes):
    """
    Makes latex-readible overlay of Atlas and CBCT
    """

    # Read image data
    images = [x for x in os.listdir(Atlas_dir) if x.endswith("tif")]
    for image in images:
        if "transformed_resliced" in image:
            Atlas = tf.imread(os.path.join(Atlas_dir, image))
            Atlas = np.einsum('ijk->ikj', Atlas)

    fig, _  = multiview_overlay(CBCT, np.ma.masked_where(Atlas == 0, Atlas),
                                cutplanes=cutplanes,
                                cmap1='gray', cmap2='Set1',
                                Title1="CBCT", Title2="Atlas",
                                show_cross=False,
                                vmin1=-500, vmax1=200)

    fig_fname = os.path.join(dst, 'Images', 'CBCT_Atlas.png')
    fig.savefig(fig_fname, dpi=150)
    plt.close(fig)
    caption = "Overlay of CBCT and registered DSURQE brain atlas\n"

    # Make Atlas.tex
    f = os.path.join(dst, "Atlas.tex")
    with open(f, 'wt') as file:
        file.write("\\section{Atlas image data}\n")
        file.write(latex_figure(fig_fname, caption=caption))
        file.write("\\FloatBarrier")
    file.close()

    # Create Brain bounding box for other methods
    Atlas[Atlas >= 1] = 1
    sag_projection = np.max(Atlas, axis= 0)
    x = find_peaks(abs(np.diff(np.max(sag_projection, axis=1))))[0]
    z = find_peaks(abs(np.diff(np.max(sag_projection, axis=0))))[0]

    ax_projection = np.max(Atlas, axis= 1)
    y = find_peaks(abs(np.diff(np.max(ax_projection, axis=1))))[0]

    boundaries = [y, x, z]

    # add 20 pixel margin to boundary
    for i in range(len(boundaries)):
        boundaries[i][0] -= 20
        boundaries[i][1] += 20

    return boundaries



def create_DoseLET(Dose_dir, CBCT, dst, cutplanes, mouse=None):
    """
    Makes latex-readible overlay of Atlas, Dose and LET
    """

    if not os.path.exists(os.path.join(Dose_dir, "Dose.tif")):
        print("Must be a control animal!")
        return 0

    dose_dict = {'P2A_B6_M1': '$85\,Gy$',
                 'P2A_B6_M2': '$65\,Gy$',
                 'P2A_B6_M6': '$45\,Gy$',
                 'P2A_B6_M10': '$0\,Gy$',
                 'P2A_C3H_M1': '$40\,Gy$',
                 'P2A_C3H_M3': '$80\,Gy$',
                 'P2A_C3H_M5': '$60\,Gy$',
                 'P2A_C3H_M8': '$0\,Gy$',
                 'P2A_C3H_M10': '$60\,Gy$'}

    # Read image data
    Dose = tf.imread(os.path.join(Dose_dir, "Dose.tif"))
    LET = tf.imread(os.path.join(Dose_dir, "LET.tif"))

    # Make Bragg-curve
    bragg_curve = np.sum(np.sum(Dose, axis=1), axis=0)

    # Mask Dose&LET
    Dose = np.ma.masked_where(Dose == 0, Dose)
    LET = np.ma.masked_where(LET == 0, LET)

    # rearrange axis
    Dose = np.einsum('ijk->jki', Dose)
    LET = np.einsum('ijk->jki', LET)

    fig_Bragg = plt.figure(figsize=(8, 8*(CBCT.shape[0]/CBCT.shape[1])))
    ax = fig_Bragg.add_subplot()

    ax.imshow(CBCT[:,:, cutplanes[2]], cmap = 'gray', vmin=-500, vmax=200)
    ax.set_ylim(CBCT.shape[0], 0)
    ax.axis('off')

    ax2 = ax.twinx()
    ax2.plot(bragg_curve/np.max(bragg_curve), 'orange', linewidth=4,
             label = 'relative Dose')
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel('D/$D_{max}$')
    ax2.legend()
    fig_Bragg.tight_layout()

    fig1_dose, _ = multiview_overlay(CBCT, Dose, cutplanes=cutplanes,
                                     cmap1='gray', cmap2='inferno',
                                     Title1="CBCT", Title2="Dose",
                                     show_cross=True, alpha=0.6,
                                     vmin1=-500, vmax1=200)
    fig_LET, _ = multiview_overlay(CBCT, LET, cutplanes=cutplanes,
                                   cmap1='gray', cmap2='inferno',
                                   Title1="CBCT", Title2="LET",
                                   show_cross=True, alpha=0.6,
                                   vmin1=-500, vmax1=200)

    fname_Bragg = os.path.join(dst, 'Images', 'Bragg_Curve.png')
    fig_Bragg.savefig(fname_Bragg, dpi=150)
    caption_Bragg = latexify('Axial CBCT slice with overlaid summed and ' +
                             'normalized proton dose (orange).')
    plt.close(fname_Bragg)

    fig_fname_dose = os.path.join(dst, 'Images', 'CBCT_Dose.png')
    fig1_dose.savefig(fig_fname_dose, dpi=150)
    plt.close(fig1_dose)

    fig_fname_LET = os.path.join(dst, 'Images', 'CBCT_LET.png')
    fig_LET.savefig(fig_fname_LET, dpi=150)
    plt.close(fig_LET)
    caption_LET = "Overlay of CBCT and simulated LET distribution\n"
    caption_dose = "Overlay of CBCT and simulated dose distribution\n"

    # Make Simulation.tex
    f = os.path.join(dst, "Simulation.tex")
    with open(f, 'wt') as file:
        file.write("\\section{Simulation image data}\n")
        file.write('Received dose (mean dose within $0.8\\cdot D_{max}$): ' +
                   latexify(dose_dict[mouse]))
        file.write(latex_figure(fig_fname_dose, caption=caption_dose))
        file.write(latex_figure(fig_fname_LET, caption=caption_LET))
        file.write(latex_figure(fname_Bragg, caption=caption_Bragg))
        file.write("\\FloatBarrier")


def S2N(Array, lower_perc=0.05, upper_perc=0.95):
    """
    Measure the signal-to-noise ratio in an nd-array.
    The Signal is defined as the signal of the upper percentile of the image
    (default = 0.95) divided by the standard deviation of background pixels
    that are defined as pixels with gray values smaller than the lower percentile
    (default = 0.05)
    """

    Array = (Array - np.min(Array))/(np.max(Array) - np.min(Array))

    Noise = np.std(Array[Array < np.quantile(Array, lower_perc)].flatten())
    Signal = np.quantile(Array, upper_perc)

    return Signal/Noise

def create_MRI(MRI_dir, CBCT, dst, cutplanes, boundaries):

    sequences = ["T1", "T2"]
    timepoints = [x for x in os.listdir(MRI_dir) if "Week" in x]

    f = os.path.join(dst, "MRI.tex")

    if os.path.exists(f):
        os.remove(f)

    # Create file handle
    file = open(f,'wt')
    file.write("\\newpage\n")

# =============================================================================
#     # First: Compilation of raw & denoised
# =============================================================================
    file.write("\\section{MRI image data}\n")
    file.write("\\subsection{{Denoising}}\n")

    for seq in sequences:

        # lists for signal to noise values
        raw = []
        deN = []
        outp = []
        Delta_SNR = []

        file.write("\\subsubsection{{Sequence type: {:s}}}\n".format(seq))

        for tp in tqdm(timepoints, desc='Denoising ' + seq):
            directories = os.listdir(os.path.join(MRI_dir, tp))

            # Identify processing steps
            dir_deN = [x for x in directories if "denoised" in x][0]
            dir_deN = os.path.join(MRI_dir, tp, dir_deN)
            dir_raw = dir_deN.replace("_denoised", "")

            f = [x for x in os.listdir(dir_raw) if seq in x and "dcm" in x]

            if len(f) == 0:
                continue
            else:
                f=f[0]
            Img_raw  = dcm.read_file(os.path.join(dir_raw, f)).pixel_array
            Img_raw = np.einsum('ijk->jki', Img_raw)

            # measure S2N-ratio
            S2Nr_raw = S2N(Img_raw)

            # Plot raw image and get bounding box
            fig, ax = plt.subplots(nrows=2, ncols=3)
            bds = get_bounding_box(Img_raw, margin=20, perc=0.5)
            multiview_overlay(Img_raw, cmap1='gray', bds=bds, ax=ax[0,:])

            f = [x for x in os.listdir(dir_deN) if seq in x and "tif" in x]
            if len(f) == 0:
                continue
            else:
                f=f[0]
            Img_deN  = tf.imread(os.path.join(dir_deN, f))
            Img_deN = np.einsum('ijk->jki', Img_deN).astype('uint16')
            S2Nr_deN = S2N(Img_deN)  # calculate signal to noise ratio

            multiview_overlay(Img_deN, cmap1='gray', bds=bds, ax=ax[1,:])

            [x.tick_params(labelleft=False, labelbottom=False) for x in ax.flatten()]
            ax[0, 0].set_ylabel('Raw MRI', fontsize = 14)
            ax[1, 0].set_ylabel('Denoised MRI', fontsize = 14)
            ax[0, 0].set_title('Coronal')
            ax[0, 1].set_title('Sagittal')
            ax[0, 2].set_title('Axial')

            # Save image
            fig_fname_deN = os.path.join(dst, 'Images', "MRI_" + seq + "_" + tp + "_denoised.png")
            fig.savefig(fig_fname_deN, dpi=150)
            plt.close(fig)

            # To tex
            caption = latexify(', '.join(["Raw and denoised MRI image",
                                          "Timepoint:" + tp,
                                          "Sequence type: " + seq,
                                          "Signal-to-noise ratio (raw)$={:.2f}$".format(S2Nr_raw),
                                          "Signal-to-Noise ratio (denoised) $= {:.2f}$".format(S2Nr_deN)]))
            file.write(latex_figure(fig_fname_deN, caption=caption))
            raw.append(S2Nr_raw)
            deN.append(S2Nr_deN)
            Delta_SNR.append(S2Nr_deN - S2Nr_raw)
        file.write("\\FloatBarrier\n")
        outp.append("\\noindent\n")
        outp.append("$SNR_{{raw, {:s}}} = {:.2f} \\pm {:.2f}$".format(
                seq, np.mean(raw), np.std(raw)) + "\\newline\n")
        outp.append("$SNR_{{denoised, {:s}}} = {:.2f} \\pm {:.2f}$".format(
                seq, np.mean(deN), np.std(deN)) + "\\newline\n")
        test_res = ttest_ind(deN, raw, alternative='greater')
        outp.append("$SNR_{{denoised, {:s}}} > SNR_{{raw, {:s}}}, ".format(
                seq, seq) + " (p< {:.1e})$\\newline\n".format(test_res.pvalue))
        outp.append("$\\Delta SNR = {:.2f} \\pm {:.2f}$\n".format(
            np.mean(Delta_SNR), np.std(Delta_SNR)))

    # file.write("\\subsubsection{{Signal-to-noise}}")
        [file.write(out) for out in outp]


# =============================================================================
#     # Next display results from registration of timepoints
# =============================================================================
    file.write("\\subsection{{Registration (timepoints)}}\n")
    for seq in sequences:

        file.write("\\subsubsection{{Sequence type: {:s}}}\n".format(seq))

        fig, ax = plt.subplots(nrows=3, ncols=len(timepoints))
        [x.axis('off') for x in np.ravel(ax)]
        first_week = True

        # First: Compilation of raw & denoised
        for i, tp in enumerate(tqdm(timepoints, desc='LOngitudinal reg.' + seq)):
            directories = os.listdir(os.path.join(MRI_dir, tp))

            dir_reg = [x for x in directories if "registered" in x][0]
            dir_reg = os.path.join(MRI_dir, tp, dir_reg)
            f = [x for x in os.listdir(dir_reg) if seq in x and "tif" in x]

            if len(f) == 0:
                [x.axis('off') for x in np.ravel(ax[:, i])]
                ax[0, i].set_title("No data")
                continue
            else:
                f=f[0]

            Img_reg  = tf.imread(os.path.join(dir_reg, f))
            Img_reg = np.einsum('ijk->jki', Img_reg).astype('uint16')
            Img_reg = (Img_reg - Img_reg.min())/(Img_reg.max() - Img_reg.min())

            # get bounding rectangle and cutplanes from first timepoint
            if first_week:
                bds = get_bounding_box(Img_reg, margin=20, perc=0.5)
                Img_reg = Img_reg[bds[0][0] : bds[0][1],
                                  bds[1][0] : bds[1][1],
                                  bds[2][0] : bds[2][1]]
                cp = np.asarray(ndimage.center_of_mass(Img_reg)).astype(int)
                Summed_image = np.zeros_like(Img_reg)

                first_week = False
            else:
                Img_reg = Img_reg[bds[0][0] : bds[0][1],
                                  bds[1][0] : bds[1][1],
                                  bds[2][0] : bds[2][1]]

            _min = np.quantile(Img_reg.flatten(), 0.5)
            _max = Img_reg.max()
            ax[0, i].imshow(Img_reg[cp[0], :, :], cmap='gray', vmin=_min, vmax=_max)
            ax[1, i].imshow(Img_reg[:, cp[1], :], cmap='gray', vmin=_min, vmax=_max)
            ax[2, i].imshow(Img_reg[:, :, cp[2]], cmap='gray', vmin=_min, vmax=_max)
            ax[0, i].set_title(tp.replace("_", "\n"))

            ax[0, 0].set_ylabel("Coronal")
            ax[1, 0].set_ylabel("Sagittal")
            ax[2, 0].set_ylabel("Axial")

            Summed_image += Img_reg


        # Image stuff
        [fig.tight_layout() for x in range(int(i/2))]
        fig_fname_reg = os.path.join(dst, 'Images', "MRI_" + seq + "_" +
                                     "_registered_gallery.png")
        fig.savefig(fig_fname_reg, dpi=150)
        plt.close(fig)

        # Summed overview
        fig_sum, _ = multiview_overlay(Summed_image, cmap1='gray',
                                       Title1="Summed overlay", show_cross=True,
                                       vmin1=np.quantile(Summed_image, 0.1),
                                       vmax1=Summed_image.max())
        fig_fname_sum = os.path.join(dst, 'Images', "MRI_" + seq + "_registered_overlay.png")
        fig_sum.savefig(fig_fname_sum, dpi=150)
        plt.close(fig_sum)

        # To LaTeX
        caption1 = latexify(', '.join(["Gallery of registered MRI images ",
                                      "Sequence type: " + seq]))
        caption2 = latexify(', '.join(["Summed overlay of registered MRI images ",
                                      "Sequence type: " + seq]))

        file.write(latex_figure(fig_fname_reg, caption=caption1, width=1.0))
        file.write(latex_figure(fig_fname_sum, caption=caption2, width=1.0))
        file.write("\\FloatBarrier")


# =============================================================================
#     # Next display results from warping to CT
# =============================================================================
    file.write("\\begin{landscape}\n")
    file.write("\\subsection{{Warping to CBCT}}\n")

    for seq in sequences:

        fig, ax = plt.subplots(nrows=3, ncols=len(timepoints), figsize=(16,4))
        # [x.tick_params(left=False, )  # remove the ticks for x in np.ravel(ax)]
        [x.axis('off') for x in np.ravel(ax)]

        # First: Compilation of raw & denoised
        for i, tp in enumerate(tqdm(timepoints, desc='Warping:' + seq)):
            directories = os.listdir(os.path.join(MRI_dir, tp))

            dir_reg = [x for x in directories if "warped" in x][0]
            dir_reg = os.path.join(MRI_dir, tp, dir_reg)
            f = [x for x in os.listdir(dir_reg) if seq in x and "tif" in x]

            cp = cutplanes
            ax[0, i].imshow(CBCT[cp[0], :, :], cmap='gray', vmin=-300, vmax=400)
            ax[1, i].imshow(CBCT[:, cp[1], :], cmap='gray', vmin=-300, vmax=400)
            ax[2, i].imshow(CBCT[:, :, cp[2]], cmap='gray', vmin=-300, vmax=400)

            if len(f) == 0:
                [x.axis('off') for x in np.ravel(ax[:, i])]
                ax[0, i].set_title("No MRI data")
                continue
            else:
                f = f[0]

            if "csv" in f:
                continue

            Img_warp = tf.imread(os.path.join(dir_reg, f))
            Img_warp = np.einsum('ijk->jki', Img_warp).astype('uint16')
            _min = np.quantile(Img_warp.flatten(), 0.1)
            _max = Img_warp.max()
            Img_warp = np.ma.masked_where(Img_warp < 30000, Img_warp)

            ax[0, i].imshow(Img_warp[cp[0], :, :], cmap='Oranges_r', vmin=33300, vmax=_max, alpha=0.6)
            ax[1, i].imshow(Img_warp[:, cp[1], :], cmap='Oranges_r', vmin=33300, vmax=_max, alpha=0.6)
            ax[2, i].imshow(Img_warp[:, :, cp[2]], cmap='Oranges_r', vmin=33300, vmax=_max, alpha=0.6)

            ax[0, i].set_title(tp.replace("_", "\n"))

        # Image formatting
        [
            (x.axis('on'), x.set_xticklabels([]), x.set_yticklabels([]))
            for x in np.ravel(ax[:, 0])
        ]
        # [x.axis('on') for x in np.ravel(ax[:, 0])]
        ax[0, 0].set_ylabel("Coronal")
        ax[1, 0].set_ylabel("Sagittal")
        ax[2, 0].set_ylabel("Axial")
        [fig.tight_layout() for x in range(int(i/2))]

        # Saving and writting to TeX
        fig_fname_warp = os.path.join(dst, 'Images', "MRI_" + seq + "_" + "_warped_gallery.png")
        fig.savefig(fig_fname_warp, dpi=150)
        plt.close(fig)

        caption = latexify(', '.join(["Gallery of CBCT (gray) and warped "
                                      "MRI images (orange) ",
                                      "Sequence type: " + seq]))
        file.write(latex_figure(fig_fname_warp, caption=caption))
    file.write("\\end{landscape}\n")
    file.close()

def preprocess(Image):
    """
    Processes histological images:
        - histogram correction: Elastix sometimes warps elements > 2**16
            into the negative space.
        - Normalization to [0..1]
    """
    Image = np.where(Image >= 0, Image, Image + 2**16)
    return (Image - np.min(Image))/(np.max(Image) - np.min(Image))

def create_Histology(Histo_dir, CBCT, dst, boundaries, qt=0.9,
                     bundles=True, S2V=True, close_up=True):

    Slice_dir = os.path.join(Histo_dir, "Slices")
    Slice_dirs = os.listdir(Slice_dir)

    # Make directory for figures
    fig_dir = os.path.join(dst, 'Images', 'Hist')
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # Generate tex file
    f = os.path.join(dst, "Histology.tex")
    if os.path.exists(f):
        os.remove(f)

    # Write header
    file = open(f,'wt')
    file.write("\\newpage\n")
    file.write("\\section{Histology image data}\n")
    file.write("\\subsection{Bundle registration}\n")

    file.write("Number of section locations:{:d}\\\\ \n".format(len(Slice_dirs)))
    file.write("Stainings for this sample:\n")
    imagetypes = []
    for base, subdirs, files in os.walk(Slice_dir):
        for f in files:
            if "czi" in f and "NOK" in f:
                imagetypes.append("NOK")

            if "czi" in f and "HE" in f:
                imagetypes.append("HE")

            if "czi" in f and "GNI" in f:
                imagetypes.append("GNI")

            if "czi" in f and "GIN" in f:
                imagetypes.append("GIN")

            if "czi" in f and "NC" in f:
                imagetypes.append("NC")

            if "czi" in f and "KHM" in f:
                imagetypes.append("KHM")

            if "czi" in f and "IC" in f:
                imagetypes.append("IC")

            if "czi" in f and "NGO" in f:
                imagetypes.append("NGO")

    imagetypes = list(set(imagetypes))

    # rearrange list so that HE is first entry
    if 'HE' in imagetypes and imagetypes[0] != 'HE':
        i = [i for i, x in enumerate(imagetypes) if x == 'HE'][0]
        imagetypes[0], imagetypes[i] = imagetypes[i], imagetypes[0]
    image_desc = []
    for i, it in enumerate(imagetypes):
        if it == "HE":
            image_desc.append("HE: Hematoxylin \\& eosin staining")
        if it == "GNI":
            image_desc.append( "GNI: GFAP, Nestin, Iba1 and DAPI")
        if it == "GIN":
            image_desc.append( "GNI: GFAP, Iba1, Nestin and DAPI")
        if it == "NOK":
            image_desc.append("NOK: NeuN, OSP, Ki-67 and DAPI")
        if it == "NGO":
            image_desc.append("NGO: NeuN, GFAP, OSP and DAPI")
        if it == "KHM":
            image_desc.append("KHM: Ki-67, Hif1a, MAP2 and DAPI")
        if it == "NC":
            image_desc.append("NC: Nestin, Cas3 and DAPI")
        if it == "IC":
            image_desc.append("IC: Iba1, CD45 and DAPI")
    file.write(itemize(image_desc))

    file.write("\\textbf{Note}: Pairs of histochemical (e.g. HE) and immunofluorescent (e.g. GNI)"
               " images are registered with a contour-based technique, whereas"
               " pairs of immunofluorescent images (e.g. GNI and NOK) are"
               " registered with an intensity-based method. For the latter,"
               " the DAPI canal present in all immunofluorescent stainings"
               " serves as registration reference and is displayed in this"
               " overview.\n")

# =============================================================================
#   Bundle registrations
# =============================================================================

    if bundles:
        # Allocate QA parameters
        Q_matrix = np.zeros((len(imagetypes), len(Slice_dirs)))

        for s, slc_dir in enumerate(tqdm(Slice_dirs, desc="bundle registrations")):
            path = os.path.join(Slice_dir, slc_dir)
            file.write("\\subsubsection{{{:s}}}\n".format(latexify(slc_dir)))

            # Write raw image data is present
            file.write("Raw images:\\\\\n")
            file.write(itemize([x for x in os.listdir(path) if "czi" in x]))

            # find directories with results from mask registration
            path_res = os.path.join(path, "result")
            mask_results = [x for x in os.listdir(path_res) if not "."  in x]
            file.write("Registered stainings: {:d}/{:d}\n".format(len(mask_results)+1, i+1))

            # Go through masks of bundle registrations
            for m_r in mask_results:
                MovingMask = tf.imread(os.path.join(path_res, m_r, "MovingMask.tif"))
                TargetMask = tf.imread(os.path.join(path_res, m_r, "TargetMask.tif"))
                result = tf.imread(os.path.join(path_res, m_r, "result.tif"))

                staining_m = m_r.split('_to_')[0]
                staining_m = [(x, i) for i, x in enumerate(imagetypes) if x in staining_m][0]

                # Calculate Jaccard index or mutual information
                if np.max(TargetMask) == 255:
                    U = result + TargetMask
                    I = np.multiply(result, TargetMask)

                    U[U != 0] = 1
                    I[I != 0] = 1

                    metric = np.sum(I)/np.sum(U)


                    caption=latexify("Results of registration of "
                                      "image {:s} (Moving image) to {:s} "
                                      "(Target image). ".format(m_r.split("_to_")[0],
                                                                m_r.split("_to_")[1]) +
                                      "Jaccard-coefficient $= {:.2f}$".format(metric))
                else:
                    MovingMask = preprocess(MovingMask)
                    TargetMask = preprocess(TargetMask)
                    result = preprocess(result)
                    metric = MutualInformation(result, TargetMask,
                                                sample_size=50000)
                    # metric2 = MutualInformation2(result, TargetMask)
                    caption=latexify("Results of registration of "
                                      "image {:s} (Moving image) to {:s} "
                                      "(Target image). ".format(m_r.split("_to_")[0],
                                                              m_r.split("_to_")[1]) +
                                      "$MI_{{norm}}$ = {:.2f}".format(metric))
                Q_matrix[staining_m[1], s] = metric

                MovingMask = np.ma.masked_where(MovingMask == 0, MovingMask)
                TargetMask = np.ma.masked_where(TargetMask == 0, TargetMask)
                result = np.ma.masked_where(result == 0, result)

                fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(12,3))

                ax[0].imshow(MovingMask, cmap='Blues', vmin=0, vmax=np.quantile(MovingMask, qt))
                ax[1].imshow(result, cmap='Blues', vmin=0, vmax=np.quantile(result, qt))
                ax[2].imshow(TargetMask, cmap='Oranges', vmin=0, vmax=np.quantile(TargetMask, qt))
                ax[3].imshow(TargetMask, cmap='Oranges', vmin=0, vmax=np.quantile(TargetMask, qt), alpha=0.5)
                ax[3].imshow(result, cmap='Blues', vmin=0, vmax=np.quantile(result, qt), alpha=0.5)

                ax[0].set_title("Moving image")
                ax[1].set_title("Transf. image")
                ax[2].set_title("Target image")
                ax[3].set_title("Overlaid images")
                fig.tight_layout()

                fig_fname_mask = os.path.join(fig_dir, slc_dir + "_" + m_r + ".png")
                fig.savefig(fig_fname_mask, dpi=150)

                file.write(latex_figure(fig_fname_mask, caption=caption, subdir="Hist/"))
                plt.close(fig)
            file.write("\\FloatBarrier\n")

        redundant_row = np.where(np.sum(Q_matrix, axis=1) == 0)[0][0]
        Q_matrix = np.delete(Q_matrix, (redundant_row), axis=0)
        IT = copy.deepcopy(imagetypes)
        IT.remove(IT[redundant_row])

        Q_matrix[Q_matrix == 0]  = np.nan

        # Make a plot for JC and MI scale indication. This is gonna look cool!
        cmap = make_colormap([c('blue'), c('orange')])
        cmap.set_bad('gray')
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))

        # JC
        img = axes[0].imshow(Q_matrix[0,:][np.newaxis, ...], cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(img, orientation='horizontal', label='Jaccard index', ax=axes[0])
        axes[0].set_yticks([0])
        axes[0].set_yticklabels(['HE'])

        # MI
        img = axes[1].imshow(Q_matrix[1:,:], cmap = cmap, vmin=0, vmax=1)
        plt.colorbar(img, orientation='horizontal', label='Mutual information', ax=axes[1])
        axes[1].set_yticks(np.arange(0, len(IT)-1, 1))
        axes[1].set_yticklabels(IT[1:])

        for ax in axes:
            ax.set_xticks(np.arange(0, len(Slice_dirs), 1))
            ax.set_xticklabels(Slice_dirs, rotation =45, ha="left")
            ax.xaxis.set_ticks_position('top')

        fig.tight_layout()
        fname_fig_QA = os.path.join(fig_dir, "Quality_matrix.png")
        fig.savefig(fname_fig_QA, dpi=150)
        plt.close(fig)

        caption = latexify("Separate Jaccard index (JC) and normalized mutual information (MI) "
                            "for every histology bundle and registration step.")
        file.write(latex_figure(fname_fig_QA, subdir="Hist/", caption=caption))


        file.write("Overall quantified intensity-based registration quality:\\newline\n")
        file.write("Normalized mutual information for intensity-based registrations $MI_{{norm}} = {:.2f} \\pm {:.2f}$, ".format(
                np.nanmean(Q_matrix[1:, :]), np.nanstd(Q_matrix[1:, :])))
        file.write("Jaccard-coefficient (JC) for contour-based registrations "
                    "$JC = {:.2f}\\pm{:.2f}$\\newline\n".format(
                        np.nanmean(Q_matrix[0, :]), np.nanstd(Q_matrix[0, :])))


# =============================================================================
#     Slice2Volume
# =============================================================================

    if S2V:
        N = len(Slice_dirs)
        file.write("\\subsection{Slice2Volume}\n")
        S2V_dir_res = os.path.join(Histo_dir, "Warped", "results")
        S2V_dir_trf = os.path.join(Histo_dir, "Warped", "trafo")

        n_inv = 0
        n_fwd = 0
        for f in os.listdir(S2V_dir_trf):
            if "inverse" in f and f.endswith('txt'):
                n_inv += 1
            elif not "inverse" in f and f.endswith('txt'):
                n_fwd +=1

        file.write("Forward transformation files: {:d}/{:d}\\\\\n".format(n_fwd, N))
        file.write("Inverse transformation files: {:d}/{:d}\\\\\n".format(n_inv, N))

        # Print S2V config parameters
        # f_log = os.path.join(S2V_dir_res, "S2V_LogFile.txt")
        # if os.path.exists(f_log):
        #     file.write("Used S2V configuration:\\newline\n")
        #     file.write(txt2latex(f_log))


        # Overlay with CT data
        for f in os.listdir(S2V_dir_res):
            if "OutputStack_interpolated_" in f:
                trnsfd_ch = f.replace("OutputStack_interpolated_", "").split(".")[0]
                break

        DAPI = tf.imread(os.path.join(S2V_dir_res, f))
        DAPI = np.einsum('ijk->ikj', DAPI)


        # sequential DAPI figure to check wrong order of slices - should show here
        bds = boundaries

        # find number of first and last non-zero ccoronal slice here
        idx  = 0
        flag = True
        while True:
            idx += 1
            img = DAPI[bds[0][0] + idx, bds[1][0]:bds[1][1], bds[2][0]:bds[2][1]]
            if np.max(img) > 30 and flag:
                y0 = bds[0][0] + idx
                flag = False

            if np.max(img) == 0 and not flag:
                y1 = bds[0][0] + idx
                break

        slcs = int(np.sqrt(y1 -y0) + 0.5)
        fig0, axes = plt.subplots(nrows=slcs, ncols=slcs)

        for i, ax in enumerate(np.ravel(axes)):
            img = DAPI[y0 + i, bds[1][0]:bds[1][1], bds[2][0]:bds[2][1]]
            ax.imshow(img, cmap='inferno', vmax=3000)
            ax.axis('off')
        fig0.tight_layout()
        fig_sequential_fname = os.path.join(fig_dir, "S2V_sequential.png")
        fig0.savefig(fig_sequential_fname, dpi=150)

        caption=("Sequence of coronal DAPI images that were transformed "
                  "with Slice2Volume. This view allows to evaluate the "
                  "consistency of subsequent slices.")
        file.write(latex_figure(fig_sequential_fname, subdir="Hist/", caption=caption))
        plt.close(fig0)

        DAPI = np.ma.masked_where(DAPI == 0, DAPI)
        fig, args  = multiview_overlay(CBCT, DAPI, cmap1='gray', cmap2='jet',
                                        Title1="CBCT", Title2="DAPI", show_cross=True,
                                        alpha=0.7, vmin1=-500, vmax1=300,
                                        vmin2 = 20, vmax2=1500)
        fig_fname = os.path.join(fig_dir, 'S2V_overlay.png')
        fig.savefig(fig_fname, dpi=150)
        plt.close(fig)


        caption = ("Overlay of downsampled and transformed histological image "
                    "({:s}) and CBCT".format(latexify(trnsfd_ch)))
        file.write(latex_figure(fig_fname, subdir="Hist/", caption=caption))
        file.write('\\FloatBarrier\n')

# =============================================================================
#     Close-up staining collage
# =============================================================================

    if close_up:
        file.write("\\subsection{Histology close-up}\n")

        # go through slice assignment overview until a slice with all stainings
        offset = 0
        while True:

            # go through S2V slice assignment overview file
            with open (os.path.join(Histo_dir, 'Warped', 'results',
                                    'SliceAssignment_Overview.txt')) as S2V_out:

                for row in reversed(list(S2V_out)):
                    info = row.split('\t')
                    slc = int(info[-1][:-1])
                    plane = os.path.normpath(info[1]).split('\\')  # extract image location (XXXX_Scene_Y)
                    if slc <= args['cutplanes'][0] - offset:
                        break

            # Parse plane from text file
            plane = [x for x in plane if "Scene" in x][0]

            # Find all raw images
            dir_raw = os.path.join(Histo_dir, 'Slices', plane)
            imgs = [x for x in os.listdir(dir_raw) if x.endswith('czi')]

            if len(imgs) != len(imagetypes):
                offset = offset + 1
                continue

            # Find all tifs
            dir_images = os.path.join(Histo_dir, 'Slices', plane, 'result')
            imgs = [x for x in os.listdir(dir_images) if x.endswith('tif')]
            imgs = [x for x in imgs if 'transformed_DAPI' not in x]  # remove transformed DAPI

            # check if HE is in list of images
            if any(['HE' in x for x in imgs]):
                L = len(imgs) - 2
                print('Making close-up for section ' + plane)
            else:
                L = 0
                offset = offset + 1

            if L != 0:
                break

        # make random 1000x1000 ROIs in DAPI image
        DAPI = tf.imread(os.path.join(dir_images, [x for x in imgs if 'DAPI' in x][0]))

        # ROI locations
        size = 1000
        locations = [(DAPI.shape[1]//3, DAPI.shape[0]//2),  #(y, x)
                     (2*DAPI.shape[1]//3, DAPI.shape[0]//2)]  #(y, x)

        # create plots for close up views
        fig_close_up = []
        for loc in locations:
            W = 6
            H = int(np.ceil(L/4))

            fig = plt.figure(figsize=(W*3, H*3))
            gs = fig.add_gridspec(H, W)
            fig_close_up.append(fig)

            # Show DAPI overview:
            rect = Rectangle((loc[1], loc[0]), size, size,
                             linewidth=1, edgecolor='r', facecolor='none')
            ax = fig.add_subplot(gs[:, :2])
            cmap, vmin, vmax = get_cmap('DAPI', DAPI[::4, ::4])
            ax.imshow(DAPI, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.add_patch(rect)
            ax.axis('off')
            ax.set_title('DAPI: whole brain')

        # iterate over all tif images in result folder
        i = 0
        for f_img in tqdm(imgs, desc='Showing stainings'):

            # no doubles of DAPI
            if "transformed_DAPI" in f_img:
                continue

            # combine HE into single image (not as blue/red/green channel)
            f = os.path.join(dir_images, f_img)
            if "_HE" in f_img:
                if "HE_R" in f_img:
                    R = tf.imread(f)
                    G = tf.imread(f.replace("HE_R", "HE_G"))
                    B = tf.imread(f.replace("HE_R", "HE_B"))
                    whole = np.stack([preprocess(R),
                                      preprocess(G),
                                      preprocess(B)], axis=2)
                else:
                    continue

            else:
                whole = tf.imread(f)

            # Raw name of staining type from filename
            S_name = f_img.split('.')[0].replace('transformed_', '').split('_')[0]

            # Make whole-section image for this staining
            cmap, vmin, vmax = get_cmap(f_img, whole[::4,::4])
            fig1, ax1 = plt.subplots(figsize=(15, 15*whole.shape[0]/
                                                      whole.shape[1]))
            ax1.imshow(whole[::4,::4], cmap=cmap, vmin=vmin, vmax=vmax)
            ax1.axis('off')

            # save and write to TeX
            fname_fig1 = os.path.join(fig_dir, 'Whole_{:s}.png'.format(S_name))
            fig1.savefig(fname_fig1, dpi=350)
            caption = latexify('{:s}-staining from plane {:s}'.format(S_name, plane))
            file.write(latex_figure(fname_fig1, subdir="Hist/", caption=caption))
            plt.close(fig1)

            # Make close up collages
            for n, loc in enumerate(locations):

                # Add closeups of stainings to previously created overview
                patch = whole[loc[0]:loc[0] + size, loc[1]:loc[1] + size]
                X = i%4 + 2  # +2 because DAPI whole section image is on this figure, too
                Y = i//4
                ax = fig_close_up[n].add_subplot(gs[Y, X])
                cmap, vmin, vmax = get_cmap(f_img, whole[::4,::4])
                ax.imshow(patch, cmap=cmap, vmin=vmin, vmax=vmax)
                ax.axis('off')
                ax.set_title(S_name)

            # increase location index counter for next staining
            i += 1

        for i, fig in enumerate(fig_close_up):
            fig.tight_layout()
            fig_fname = os.path.join(fig_dir, 'Histology_detail_loc{:d}.png'.format(i))
            fig.savefig(fig_fname, dpi=350)
            caption = ('Selection of image details of the various co-aligned '
                       'stainings. The position of the image detail is indicated '
                       'by the red rectangle in the whole-brain DAPI image. '
                       'The selected section for this overview was {:s}.'.format(
                           latexify(plane)))
            file.write(latex_figure(fig_fname, subdir="Hist/", caption=caption))
            plt.close(fig)

    file.close()
