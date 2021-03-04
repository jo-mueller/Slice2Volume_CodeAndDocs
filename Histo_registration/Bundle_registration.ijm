/*
 * This script performs registration of bundles of adjacent histological sections with different stainings.
 * The functionality allows for two basic mechanisms of registration: contour-based and intensity based.
 * 
 * DATA STRUCTURE: * 
 * The data is expected to be stored in a bundle structure (see also histological data at https://rodare.hzdr.de/record/801)
 * 	
 * MODIFICATIONS
 * The user has to modify the following Blueprint snippet in the Main loop (line 150ff.):
		// Blueprint
		if(IDexistsinDir(your_staining_ID, root + "result/")) {
			print("    your_staining_ID has already been registered - skipping.");
			continue;
		} else {
			RegMain(root, 	moving staining, used magnification, exponent base for image pyramind, 
							target staining, used magnification, exponent base for image pyramind, mode: can be 'conntour' or intensity');
		}
 * 
 * Further registration steps can be added as needed. 
 * 
 * HARDWARE:
 * This script uses the Imagej plugin CLIJ2 (https://clij.github.io/) - it is required to use a GPU with at least 8GB RAM.
 */


close("*");

// User dialogue
#@ String (visibility=MESSAGE, value="Elastix files and parameters", required=false) a
#@ File (label="Elastix directory", style="directory") elastix_dir
#@ File (label="Elastix parameter file", style="file") p_file
#@ File (label="Data root directory", style="directory") basepath

#@ String (visibility=MESSAGE, value="Processing parameters", required=false) b
#@ Float (label = "Magnification level for registration", value=2.5, required=true) reg_magnification_level
#@ Float (label = "Output magnification level", value=10, required=true) out_magnification_level
#@ Integer (label = "Contour smoothing degree", value=20, required=true) r_smoothing
#@ Boolean (label = "Batch mode?", value=true) use_batch

// config
if (use_batch) {setBatchMode(true);}  // batch mode on/off
run("CLIJ2 Macro Extensions", "cl_device=[GeForce RTX 3060 Ti]");  //clij2 initialization

// For registration
Elastix_exe = elastix_dir + "/elastix.exe";
Transformix_exe = elastix_dir + "/transformix.exe";
basepath = basepath + "/";

// variables
MovingMask = "MovingMask";
TargetMask = "TargetMask";

// check whether these files actually exist.
paths = newArray(Elastix_exe, Transformix_exe, p_file, basepath);
checkInput(paths);
mice = getFileList(basepath);
Array.show(mice);

// iterate over all animals
for (m = 0; m < mice.length; m++) {

	// skip non-directories
	if (!File.isDirectory(basepath + mice[m])) {
		continue;
	}

	// select particular mouse, if desired
	if(mice[m] != "P2A_C3H_M3/"){
		continue;
	}

	samples = getFileList(basepath + mice[m] + "Histology/Slices/");
	Array.show(samples);

	
	// Iterate over all samples
	for(i=12; i<samples.length; i++){
		
		root = basepath + mice[m] + "Histology/Slices/" + samples[i];
		print("Basepath: " + root);

		// Check if sample data is propperly formatted: Has to be directory
		if (!File.isDirectory(root)) {
			print("    Error: Directory not found");
			continue;
		} else {
			images = getFileList(root);	
		}

		// Directories to be skipped are marked by underscore
		if (startsWith(samples[i], "_")) {
			continue;
		}

		// define types of raw stainings
		HE = "";  // HE staining
		NOK = "";  //NeuN, OSP, Ki67, DAPI
		GNI = "";  //GFAP, Nestin, Iba1, DAPI
		KHM = "";  // Ki-67, HIF1a, MAP2
		IC = ""; // Iba1, CD45
		NC = ""; // Nestin, Cas3
		NGO = ""; // NeuN, GFAP, OSP
		
		// Browse all raw images in this directory
		for(j=0; j<images.length; j++){
			// Identify types of staining
			if (matches(images[j], ".*HE.*")) {
				HE = images[j];
			}

			if (matches(images[j], ".*IC.*")) {
				IC = images[j];
			}

			if (matches(images[j], ".*KHM.*")) {
				KHM = images[j];
			}

			if (matches(images[j], ".*NC.*")) {
				NC = images[j];
			}

			if (matches(images[j], ".*NGO.*")) {
				NGO = images[j];
			}
	
			if (matches(images[j], ".*GNI.*") || matches(images[j], ".*GIN.*")) {
				GNI = images[j];
			}
	
			if (matches(images[j], ".*NOK.*")) {
				NOK = images[j];
			}
		}
		print("I Identified these source images:");
		print("   NOK = " + NOK);
		print("   HE = " + HE);
		print("   GNI = " + GNI);
		print("   IC = " + IC);
		print("   NC = " + NC);
		print("   NGO = " + NGO);
		print("   KHM = " + KHM);
		
		// Make directories for output data
		if (!File.exists(root + "tmp/")) {
			File.makeDirectory(root + "tmp/");
		}
		
		if (!File.exists(root + "result/")) {
			File.makeDirectory(root + "result/");
		}

		/*
		// Blueprint
		if(IDexistsinDir(your_staining_ID, root + "result/")) {
			print("    your_staining_ID has already been registered - skipping.");
			continue;
		} else {
			RegMain(root, 	moving staining, used magnification, exponent base for image pyramind, 
							target staining, used magnification, exponent base for image pyramind, mode: can be 'conntour' or intensity');
		}
		*/

		if(IDexistsinDir(IC, root + "result/")) {
			print("    IC has already been registered - skipping.");
			continue;
		} else {
			RegMain(root, 	IC, "10x", 2, 
							NC, "20x", 2, "intensity");
		}
		exit();
		
		/*
		// Check for previous registrations		
		if(IDexistsinDir("HE", root + "result/")) {
			RegMain(root, HE, "20x", 2, NOK, "10x", 2, "contour");
		} else {
			RegMain(root, HE, "20x", 2, NOK, "10x", 2, "contour");
		}
		*/

		/*
		// Check for previous registrations		
		if(IDexistsinDir("HE", root + "result/")) {
			print("    HE has already been registered");
			print("    Doing it again!");
			RegMain(root, HE, "20x", 2, GNI, "10x", 2, "contour");
		} else {
			RegMain(root, HE, "20x", 2, GNI, "10x", 2, "contour");
		}
		*/
		

		/*

		// Check for previous registrations		
		if(IDexistsinDir("NOK", root + "result/")) {
			print("    NOK has already been registered");
		} else {
			RegMain(root, NOK, "10x", 2, GNI, "10x", 2, "contour");
		}
		*/

		/*
		// CHeck for previous registrations		
		if(IDexistsinDir("KHM", root + "result/")) {
			print("    KHM has already been registered");
		} else {
			RegMain(root, KHM, "20x", 2, NGO, "20x", 2, "intensity");
		}

		// CHeck for previous registrations		
		if(IDexistsinDir("NC", root + "result/")) {
			print("    NC has already been registered");
		} else {
			RegMain(root, NC,  "20x", 2, NGO, "20x", 2, "intensity");
		}

		// CHeck for previous registrations		
		if(IDexistsinDir("IC", root + "result/")) {
			print("    IC has already been registered");
		} else {
			RegMain(root, IC,  "10x", 2, NGO, "20x", 2, "intensity");
		}
		*/
		/*
		// CHeck for previous registrations		
		if(IDexistsinDir("HE", root + "result/")) {
			print("    HE has already been registered");
			print("    Doing it again!");
			RegMain(root, HE,  "20x", 2, NGO, "20x", 2, "contour");
		} else {
			RegMain(root, HE,  "20x", 2, NGO, "20x", 2, "contour");
		}
		*/
		
	}
}

print("Macro finished!");

function RegMain(root, fmoving, mag_moving, pyramid_base_moving, ftarget, mag_target, pyramid_base_target, mode){
	/*
	 * Co-aligns image from path fmoving with image from path ftarget. 
	 * Mode can be 'contour' or 'intensity'.
	 * 	'Contour': Input will be masked and the binary masks will be used for registration
	 * 	'Intensity' Input will remain untouched and the last color channel (usually DAPI) is used for registration.
	 * 	'mag_moving' and 'mag_target' can be 10x, 20x, 40x, etc. The target resolution of the output is
	 * 	10x magnification level.
	 * 	'pyramind_base_moving' and 'pyramind_base_target' refer to thhe resolution scheme of the image pyramids of the moving and target image, respectively.
	 */

	// Check the inputs. Do not run registration if one of the files doesn't exist.
	if (fmoving == "" || ftarget == "") {
		print("   INFO: Missing input! Registration aborted.");
		return 0;
	} else {
		print("   INFO: Aligning " + fmoving + "(" + mag_moving + ")" + " with " + ftarget + "(" + mag_target + ")");
	}
	
	// The magnification parameter determines the output resolution.
	// convert magnification level to index of series in image pyramid
	// 10x ->1
	// 20x -> 2
	// 40x -> 3
	// etc

	// parse magnification level string
	mag_moving = parseFloat(substring(mag_moving, 0, lengthOf(mag_moving) - 1));
	mag_target = parseFloat(substring(mag_target, 0, lengthOf(mag_target) - 1));

	// Find the corresponding series in target and moving image pyramid
	series_moving = Magnification2Series(out_magnification_level, mag_moving, pyramid_base_moving)+1;
	series_target = Magnification2Series(out_magnification_level, mag_target, pyramid_base_target)+1;
	
	// File operations
	tmpdir = root + "tmp/";
	outdir = root + "result/";

	if (!File.exists(tmpdir)) {
		File.makeDirectory(tmpdir);
	}
	
	if (!File.exists(outdir)) {
		File.makeDirectory(outdir);
	}
	
	// Load
	//moving = FastCziLoad(fast_dir, root + fmoving, series_moving);
	//target = FastCziLoad(fast_dir, root + target, series_target);
	moving = load(root + fmoving, series_moving);
	target = load(root + ftarget, series_target);

	print("Processing moving: ", mag_moving + "x", "@series " + series_moving);
	print("Processing target: ", mag_target + "x", "@series " + series_target);

	// Check if images
	if (moving == 0 || target == 0) {
		return 0;
	}
	/*
	// Check if images are empty (or not)
	if (safetyCheck(moving) == 0) {
		return 0;
	}

	if (safetyCheck(target) == 0) {
		print("Something is off about the target image. Please check!\n\tPath: " + root + ftarget);
		return 0;
	}
	*/
	
	// Create and smooth masks
	Mask = Smooth(target, r_smoothing, reg_magnification_level, mode);
	rename(TargetMask);
	Mask = Smooth(moving, r_smoothing, reg_magnification_level, mode);
	rename(MovingMask);

	// Save masks to tmp dir
	save_custom(MovingMask, tmpdir + MovingMask, "close");
	save_custom(TargetMask, tmpdir + TargetMask, "close");

	// Separate raw stainings into tif images
	imgsize = Stack2Separate(target, outdir, series_target);
	Stack2Separate(moving, tmpdir, series_moving);	
	
	result_img = Register(Elastix_exe, p_file,
							tmpdir + MovingMask + ".tif",
							tmpdir + TargetMask + ".tif",
							tmpdir, outdir);
	
	// Move registration data to output directory for inntrospection
	_fmoving = split(fmoving, ".");
	_ftarget = split(ftarget, ".");
	introspection_dir = outdir + _fmoving[0] + "_to_" + _ftarget[0] + "/";
	if (!File.exists(introspection_dir)) {
		File.makeDirectory(introspection_dir);
	}
	File.copy(result_img, introspection_dir + "result.tif");
	File.copy(tmpdir + "MovingMask.tif", introspection_dir + "MovingMask.tif");
	File.copy(tmpdir + "TargetMask.tif", introspection_dir + "TargetMask.tif");
	File.copy(tmpdir + "TransformParameters.0.txt", introspection_dir + "TransformParameters_small.0.txt");

	// delete from tmp directory
	File.delete(result_img);
	File.delete(tmpdir + "MovingMask.tif");
	File.delete(tmpdir + "TargetMask.tif");

	// Now apply transform to entire input stack	
	TransformStack(	Transformix_exe, tmpdir + "TransformParameters.0.txt", 
					tmpdir, 
					outdir, 
					introspection_dir, 
					imgsize, 
					reg_magnification_level); // correct translation transform parameter by factor that reflects different resolutions
	close("*");

}

function Stack2Separate(image, outdir, series){
	// Opens a stack from a file and saves its channels separately.
	// Depending on the presence of certain identifiers (GNII/NOK/HE) in the filename,
	// the images are saved differently

	// Determine type of input stainings
	selectWindow(image);
	print("Saving " + image + "into separate images at location: " + outdir);
	if (matches(image, ".*NOK.*")) {
		labels = newArray("NeuN", "OSP", "Ki67", "DAPI");
	}

	if (matches(image, ".*KHM.*")) {
		labels = newArray("Ki67", "HIF1a", "MAP2", "DAPI");
	}

	if (matches(image, ".*IC.*")) {
		labels = newArray("Iba1", "CD45a", "DAPI");
	}

	if (matches(image, ".*NC.*")) {
		labels = newArray("Nestin", "Cas3", "DAPI");
	}

	if (matches(image, ".*NGO.*")) {
		labels = newArray("NeuN", "GFAP", "OSP", "DAPI");
	}

	if (matches(image, ".*GNI.*") || matches(image, ".*GIN.*")) {
		labels = newArray("GFAP", "Iba1", "Nestin", "DAPI");
	}
	
	if (matches(image, ".*HE.*")) {
		labels = newArray("HE_R", "HE_G", "HE_B");
	}

	n = nSlices;
	for(i = 1; i <= nSlices; i++){
		selectWindow(image);
		setSlice(i);
		run("Duplicate...", "title="+labels[i-1]+" duplicate channels="+i);
		saveAs("tif", outdir + labels[i-1]);
		close();
	}
	w = getWidth();
	h = getHeight();
	close(image);
	size = newArray(w, h);
	return size;
}

function TransformStack(transformix_exe, trafo_file, tmpdir, output_dir, introspection_dir, image_dimensions, mag_lvl){
	// call transformix on stacks

	series = Magnification2Series(reg_magnification_level, out_magnification_level, 2);
	factor = Math.pow(2, series);

	// Manipulate transformation params to change output image size
	trafo = File.openAsString(trafo_file);
	trafo = split(trafo, "\n");
	f = File.open(tmpdir + "Trafofile_altered2.txt");

	for(i = 0; i < trafo.length; i++){

		//  rewrite output image size. 
		if (matches(trafo[i], ".*Size.*")) {
			print(f, "(Size " + image_dimensions[0] + " " + image_dimensions[1] + ")");
			continue;
		}

		// write tiff instead of annoying mhd/raw
		if (matches(trafo[i], ".*ResultImageFormat.*")) {
			print(f, "(ResultImageFormat \"tif\")");
			continue;
		}

		//  adjust output pixel type (16bit please)
		if (matches(trafo[i], ".*ResultImagePixelType.*")) {
			print(f, "(ResultImagePixelType \"unsigned short\")");
			continue;
		}

		// adjust translation vectors
		if (startsWith(trafo[i], "(TransformParameters")) {
			line = substring(trafo[i], 1, lengthOf(trafo[i])-1);  // remove ()
			line = split(line, " ");  // get entries
			tx = parseFloat(line[line.length-2]);  // get x-translation
			ty = parseFloat(line[line.length-1]); // get y-translation
			print(f, "(TransformParameters " +line[1] + " " + line[2] + " " + 
											d2s(tx * factor, 6) + " " + 
											d2s(ty * factor, 6) + ")");
			continue;
		}

		// adjust center of rotation
		if (startsWith(trafo[i], "(CenterOfRotationPoint")) {
			line = substring(trafo[i], 1, lengthOf(trafo[i])-1);
			line = split(line, " ");
			x = parseFloat(line[line.length-2]);
			y = parseFloat(line[line.length-1]);
			print(f, "(CenterOfRotationPoint " + 
						d2s(x * factor, 2) + " " +
						d2s(y * factor, 2) + ")");
			continue;			
		}
		
		print(f, trafo[i]);
	}
	File.close(f);
	File.copy(tmpdir + "Trafofile_altered2.txt", introspection_dir + "TransformParameters_altered.0.txt");

	// Then, apply transform to every image
	ImgList = getFileList(tmpdir);
	for(i = 0; i < ImgList.length; i++){

		if (!endsWith(ImgList[i], "tif")) {
			continue;
		}

		if (matches(ImgList[i], ".*Mask.*")) {
			continue;
		}

		print("Transforming " + tmpdir + ImgList[i] + " with trafofile " + trafo_file + " to output dir " + output_dir);
		exec(transformix_exe + " " +  
			"-tp " + tmpdir + "Trafofile_altered2.txt " +
			"-out "+ output_dir + " " +
			"-in " + tmpdir + ImgList[i]);

		// Remove from tmp dir
		File.delete(tmpdir + ImgList[i]);

		// rewrite output in output dir to propper file format
		trg = output_dir + "transformed_" + ImgList[i];
		src = output_dir + "/result.tif";

		// remove trafo output if it already exists
		if (File.exists(trg)) {
			File.delete(trg);
		}
		File.rename(src, trg);
	}
}

function Register(elastix_exe, param_file, moving, target, tmpdir, outdir){
	// call elastix
	print(elastix_exe + " " +  
		"-p " + param_file + " " +
		"-out "+ tmpdir + " " +
		"-m " + moving + " " +
		"-f " + target);
	exec(elastix_exe + " " +  
		"-p " + param_file + " " +
		"-out "+ tmpdir + " " +
		"-m " + moving + " " +
		"-f " + target);
	resultfile = tmpdir + "/result.0.tif";
	return resultfile;
}

function save_custom(image, path, keep){
	selectWindow(image);

	if (is("Inverting LUT")) {
		run("Invert LUT");
	}
	
	saveAs("tif", path);
	rename(image);
	if (keep == "close") {
		close();
	}
}

function isEmpty(directory){
	// Check if a directory is empty

	if (!File.exists(directory)) {
		return true;
	}
	
	files = getFileList(directory);
	if (files.length == 0) {
		return true;
	} else {
		return false;
	}
}

function FastCziLoad(fast_directory, fname, series){

	/*
	 * Copy an image <fname> hat is stored on a slow drive 
	 * to a directory <fast_directory> on a faster drive (e.g. SSD)
	 * Then, proceed with bio-formats importer to load series <series> of
	 * the input image, assuming it's a czi-image.
	 */

	t0 = getTime();
	
	// first, copy image to the fast location
	src = fname;
	dst = fast_directory + "blubb.czi";
	File.copy(src, dst);

	// open image with Bio-formats and close the image copy on the fast drive
	run("Bio-Formats Importer", "open=" + dst + " series_"+d2s(series,0));
	File.delete(dst);
	
	// rename and return
	img = File.nameWithoutExtension;
	rename(img);

	dt = getTime() - t0;
	print("    Loaded image " + src + " in " + dt/1000 + "s.");

	return img;
}

function load(fname, series){
	// open images
	
	if (!File.exists(fname)) {
		return 0;
	}
	
	run("Bio-Formats Importer", "open=" + fname + " series_"+d2s(series,0));
	img = File.nameWithoutExtension;
	rename(img);
	return img;
}

/*
function safetyCheck(image){
	// basic safety check to see whether the pyramid levels were actually provided
	run("Set Measurements...", "mean display redirect=None decimal=2");
	selectWindow(image);
	run("Measure");
	mean_GV = getResult("Mean", nResults() - 1);
	if (mean_GV < 1) {
		print("\tSomething is odd about image " + image + " -> Please check!");
		return 0;
	} else {
		return 1;
	}
}
*/

function Smooth(image, radius, mag_lvl, mode){
	
	Ext.CLIJ2_clear();
	roiManager("reset");
	selectWindow(image);
	getDimensions(width, height, channels, slices, frames);
	
	// downsample to mag_lvl
	series = Magnification2Series(mag_lvl, out_magnification_level, 2.0);
	f = 1.0/Math.pow(2, series);
	run("Scale...", "x="+f+" y="+f+" z=1.0 width="+f*width+" height="+f*width+" depth="+nSlices+" interpolation=None process create");

	rename(image + "_downsampled");
	low_res_image = getTitle();

	// for HE: Do color deconvolution and proceed with Eosin channel
	// for others: Take last channel, because that's where DAPI is
	if (matches(low_res_image, ".*HE.*")) {
		run("RGB Color");
		RGB_img = getTitle();
		run("Colour Deconvolution", "vectors=H&E");
		close(RGB_img);
		close(RGB_img + "-(Colour_2)");
		close(RGB_img + "-(Colour_3)");
		
		selectWindow(RGB_img + "-(Colour_1)");		
		name = getTitle();
		rename("Mask");
		run("Invert");
		close(low_res_image);
	} else {
		selectWindow(low_res_image);
		setSlice(nSlices);
		run("Duplicate...", "title=Mask");
		name = getTitle();
		selectWindow(name);
		close(low_res_image);
	}
	sigma = getWidth() / 200.0;
	
	selectWindow("Mask");
	Ext.CLIJ2_push("Mask");

	// subtract background
	Ext.CLIJ2_convertFloat("Mask", Float);
	Ext.CLIJ2_bottomHatSphere(Float, tmp, 10, 10, 1);
	Ext.CLIJ2_gaussianBlur2D(tmp, Float, sigma, sigma);

	// normalize
	Ext.CLIJ2_getMeanOfAllPixels(Float, mean);
	Ext.CLIJ2_standardDeviationOfAllPixels(Float);
	std = getResult("StandardDeviation", nResults-1);
	Ext.CLIJ2_addImageAndScalar(Float, tmp, (-1) * mean);
	Ext.CLIJ2_multiplyImageAndScalar(Float, tmp, (100.0/std));

	// return image
	close("Mask");
	Ext.CLIJ2_pull(tmp);
	rename("Mask");
	
	// if contour mode: Use outline of sample for registration
	// if intensity mode: Use intensity for registration (i.e. skip the masking steps)
	if (mode == "contour") {
		Ext.CLIJ2_thresholdHuang(tmp, Mask);
		Ext.CLIJ2_closingBox(Mask, output, radius);
		Ext.CLIJ2_binaryFillHoles(output, Mask);
		Ext.CLIJ2_multiplyImageAndScalar(Mask, output, 255);
		Ext.CLIJ2_convertUInt8(output, uint8);
		
		
		close("Mask");
		Ext.CLIJ2_pull(uint8);
		rename("Mask");
		
		run("Set Measurements...", "area display redirect=None decimal=2");
		run("Analyze Particles...", "add");
	
		w = getWidth();
		h = getHeight();
	
		// Find largest blobb
		for (i = 0; i < roiManager("count"); i++) {
			roiManager("select", i);
			roiManager("Measure");
			S = getResult("Area", nResults()-1);
			if ( S < 0.001 * w * h){
				run("Clear");
			}
		}	
		run("Select None");
	}
	return "Mask";
}

function IDexistsinDir(string, directory){
	// Checks if a file with an identifier <string>
	// exists in a directory
	filelist = getFileList(directory);
	Array.show(filelist);
	for (l = 0; l < filelist.length; l++) {

		/*
		if (File.isDirectory(directory + filelist[l])) {
			continue;
		}
		*/
			
		if (matches(filelist[l], ".*" + string + ".*")) {
			return true;
			
		}
	}
	return false;
}


function checkInput(arrayOfPaths){
	for(i = 0; i < arrayOfPaths.length; i++){
		if (!File.exists(arrayOfPaths[i])) {
			print("Path " + arrayOfPaths[i] + " doesn't exist.  Check!");
			exit();
		}
	}
}

function Magnification2Series(mag, Mag0, base){
	// Determines the series in a pyramidal image format that corresponds to
	// a chosen magnification level <mag> (Can be 10x, 5x, 2.5x, 1.25x, etc)
	// The highest magnification level <Mag0> in the image (usually 10x or 20x) is 
	// required as input, as is the factor of downsampling (<base>) between each layer
	series = Math.log(Mag0/mag) * (1.0/Math.log(base));
	series = floor(series + 0.5);

	return series;
}
