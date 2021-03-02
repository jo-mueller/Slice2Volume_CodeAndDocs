/*
 * Script tthat allows you to select the data directory of a single mouse
 * and display all (registered) MRI images along the time axis in a 4D hyperstack
 */

close("*");

#@ File (label="Select a mouse", style="directory") mouse_dir
Seqs = newArray("T1", "T2");

if (!endsWith(mouse_dir, "MRI")) {
	mouse_dir = mouse_dir + "\\MRI";
}

// get available timepoints
timepoints = getFileList(mouse_dir);

// First iterate over sequence types:
for (s = 0; s < Seqs.length; s++) {

	// for this sequence, iterate over timepoints:
	n=0;
	cmd = "";
	for (i = 0; i < timepoints.length; i++) {
		tp_dir = mouse_dir + "\\" + timepoints[i];
		datasets = getFileList(tp_dir);
	
		// identify registered dataset (raw and registered should exist)
		for (j = 0; j < datasets.length-1; j++) {
			if (endsWith(datasets[j], "registered")) {
				break;
			}
		}
	
		// identify image dataset and open it
		datadir = datasets[j];
		image_files = getFileList(tp_dir + "\\" + datadir);

		// flag for image existence
		image_exists = false;
	
		for (j = 0; j < image_files.length; j++) {
			if (matches(image_files[j], ".*" + Seqs[s] + ".*") && endsWith(image_files[j], "tif")) {
				open(tp_dir + "\\" + datadir + "\\" + image_files[j]);
				rename(timepoints[i]);
				image_exists = true;
			}
		}

		if (image_exists) {
			cmd += " image" + d2s(n+1, 0) + "=" + timepoints[i];
			n +=1;
		}
	}
	//cmd = joinArray(timepoints);
	print(cmd);
	run("Concatenate...", "title=Sequence_" + Seqs[s] + " open " + cmd);
	run("Enhance Contrast...", "saturated=0.3 process_all use");
	saveAs("tif", mouse_dir + "/" + getTitle());

	for (i = 0; i < timepoints.length; i++) {
		close(timepoints[i]);
	}
}

function joinArray(array){
	str = "";
	for (i = 0; i < array.length; i++) {
		str = str + " image" + d2s(i+1, 0) + "=" + array[i];
	}
	return str;
}
