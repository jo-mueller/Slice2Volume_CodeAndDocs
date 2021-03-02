root = "E:/Promotion/Projects/2020_Slice2Volume/Data/";

setBatchMode(true);

// iterate over mice
mice = getFileList(root);
for (i = 0; i < mice.length; i++) {

	if (!matches(mice[i], ".*P2A_C3H_M8.*")) {
		continue;
	}

	timepoints = getFileList(root + mice[i] + "MRI/");
	

	// iterate over timepoints
	for (j = 0; j < timepoints.length; j++) {
		
		typelist = getFileList(root + mice[i] + "MRI/" + timepoints[j]);

		// iterate over types (raw/renoised/registered)
		for (k = 0; k < typelist.length; k++) {

			// find all nifti images in this directory and convert
			imglist = getFileList(root + mice[i] + "MRI/" + timepoints[j] + typelist[k]);
			for (m = 0; m < imglist.length; m++) {

				dir = root + mice[i] + "MRI/" + timepoints[j] + typelist[k];
				fname = dir + imglist[m];

				/*
				// undo up-down flip
				if (matches(fname, ".*15xsigma.*")) {
					// open & flip
					run("Bio-Formats Importer", "open=" + fname + " autoscale color_mode=Default rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT");
					image = getTitle();
					run("Flip Vertically", "stack");
					saveAs("tif", dir + File.getNameWithoutExtension(imglist[m]));
					close();
					print(dir + File.getNameWithoutExtension(imglist[m]));
					}
				// check if image is nifti
				if (endsWith(fname, "nii.gz")) {
					print(fname);

					// convert only 15x sigma-filtered image
					if (!matches(fname, ".*15xsigma.*")) {
						continue;
					}

					// open, convert and remove original
					run("Bio-Formats Importer", "open=" + fname + " autoscale color_mode=Default rois_import=[ROI manager] view=Hyperstack stack_order=XYCZT");
					image = getTitle();
					run("Flip Vertically", "stack");
					saveAs("tif", dir + File.getNameWithoutExtension(imglist[m]));
					close();
					File.delete(fname);
				}
				*/
				
			}
		}
	}
}