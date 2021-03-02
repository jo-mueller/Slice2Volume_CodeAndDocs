close("*");

r_smoothing = 20;
var reg_magnification_level = 2.5;  // registration is not performed at highest resolution level, but at a resolution defined by this parameter. 1.25 means 1.25x
var out_magnification_level = 10;  // output should be 10x-sized
var reg_width = 5000;  // all image masks will be downsample to a width of 5000

setBatchMode(false);
run("CLIJ2 Macro Extensions", "cl_device=[GeForce RTX 3060 Ti]");


MovingMask = "MovingMask";
TargetMask = "TargetMask";

// location
drive = "E:/Promotion/Projects";

// For registration
Elastix_exe = drive + "/2020_Slice2Volume/Scripts/elastix-5.0.1-win64/elastix.exe";
Transformix_exe = drive + "/2020_Slice2Volume/Scripts/elastix-5.0.1-win64/transformix.exe";
p_file = drive + "/2020_Slice2Volume/Scripts/Histo_registration/elastix_parameters.txt";
basepath = drive + "/2020_Slice2Volume/Data/";

// check whether these files actually exist.
paths = newArray(Elastix_exe, Transformix_exe, p_file, basepath);
checkInput(paths);
mice = getFileList(basepath);

// iterate over all animals
for (m = 0; m < mice.length; m++) {

	// select particular mouse
	if(mice[m] != "P2A_B6_M2/"){
		continue;
	}

	samples = getFileList(basepath + mice[m] + "Histology/Slices/");

	
	// Iterate over all slices
	for(i=14; i<samples.length; i++){
		
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
		HE = "";
		NOK = "";  //NeuN, OSP, Ki67, DAPI
		GNI = "";  //GFAP, Nestin, Iba1, DAPI
		KHM = "";  // Ki-67, HIF1a, MAP2
		IC = ""; // Iba1, CD45
		NC = ""; // Nestin, Cas3
		NGO = ""; // NeuN, GFAP, OSP
		
		
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
		
		// HE:
		if (!File.exists(root + "tmp/")) {
			File.makeDirectory(root + "tmp/");
		}
		
		if (!File.exists(root + "result/")) {
			File.makeDirectory(root + "result/");
		}

		

		// Check for previous registrations		
		if(IDexistsinDir("HE", root + "result/")) {
			print("    HE has already been registered");
			print("    Doing it again!");
			RegMain(root, HE, "20x", 2, GNI, "10x", 2, "intensity");
			exit();
		} else {
			RegMain(root, HE, "20x", 2, GNI, "10x", 2, "contour");
		}

		// Check for previous registrations		
		if(IDexistsinDir("NOK", root + "result/")) {
			print("    NOK has already been registered");
		} else {
			RegMain(root, NOK, "10x", 2, GNI, "10x", 2, "contour");
		}

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

		// CHeck for previous registrations		
		if(IDexistsinDir("HE", root + "result/")) {
			print("    HE has already been registered");
		} else {
			RegMain(root, HE,  "10x", 2, NGO, "20x", 2, "contour");
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

	// Move registration data to output directory for inntrospection
	_fmoving = split(fmoving, ".");
	_ftarget = split(ftarget, ".");
	introspection_dir = outdir + _fmoving[0] + "_to_" + _ftarget[0] + "/";
	if (!File.exists(introspection_dir)) {
		File.makeDirectory(introspection_dir);
	}
	
	// Create and smooth masks
	Smooth(moving, r_smoothing, reg_magnification_level, mode);
	rename(MovingMask);
	Smooth(target, r_smoothing, reg_magnification_level, mode);
	rename(TargetMask);

	// Save masks to tmp dir
	save_custom(MovingMask, tmpdir, "keep", "nrrd");
	save_custom(TargetMask, tmpdir, "keep", "nrrd");
	save_custom(MovingMask, introspection_dir, "close", "tif");
	save_custom(TargetMask, introspection_dir, "close", "tif");

	// Do registration of masks
	result_img = Register(Elastix_exe, p_file,
							tmpdir + MovingMask + ".nrrd",
							tmpdir + TargetMask + ".nrrd",
							tmpdir, outdir);
	
	
	File.copy(result_img, introspection_dir + "result.tif");
	File.copy(tmpdir + "MovingMask.tif", introspection_dir + "MovingMask.tif");
	File.copy(tmpdir + "TargetMask.tif", introspection_dir + "TargetMask.tif");
	File.copy(tmpdir + "TransformParameters.0.txt", introspection_dir + "TransformParameters_small.0.txt");

	// delete from tmp directory
	File.delete(result_img);
	File.delete(tmpdir + "MovingMask.tif");
	File.delete(tmpdir + "TargetMask.tif");

	// Separate raw stainings into tif images
	imgsize = Stack2Separate(target, outdir, series_target, "tif");
	Stack2Separate(moving, tmpdir, series_moving, "nrrd");	

	// Now apply transform to entire input stack	
	TransformStack(	Transformix_exe, tmpdir + "TransformParameters.0.txt", 
					tmpdir, 
					outdir, 
					introspection_dir,
					imgsize);
	close("*");

}

function Stack2Separate(image, outdir, series, output_type){
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
		if (output_type == "tif") {
			saveAs("tif", outdir + labels[i-1]);
		}
		if (output_type == "nrrd") {
			run("Nrrd ... ", "nrrd= "+ outdir + labels[i-1] +".nrrd");
		}
		run("Duplicate...", "title="+labels[i-1]+" duplicate channels="+i);
		
		close();
	}
	w = getWidth();
	h = getHeight();
	close(image);
	size = newArray(w, h);
	return size;
}

function TransformStack(transformix_exe, trafo_file, tmpdir, output_dir, introspection_dir, image_dimensions){

	// Alter trafo file to adjust for different image size
		// Manipulate transformation params to change output image size
	trafo = File.openAsString(trafo_file);
	trafo = split(trafo, "\n");
	f = File.open(tmpdir + "Trafofile_altered.txt");

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
		print(f, trafo[i]);
	}

	File.close(f);
	File.copy(tmpdir + "Trafofile_altered.txt", introspection_dir + "TransformParameters_altered.0.txt");
	
	// call transformix on stacks
	
	ImgList = getFileList(tmpdir);
	for(i = 0; i < ImgList.length; i++){

		if (!endsWith(ImgList[i], "tif")) {
			continue;
		}

		if (matches(ImgList[i], ".*Mask.*")) {
			continue;
		}

		print("Transforming " +ImgList[i] + "\n    with " + trafo_file + "\n    to output dir " + output_dir);
		exec(transformix_exe + " " +  
			"-tp " + tmpdir + "Trafofile_altered.txt" +
			"-out "+ output_dir + " " +
			"-in " + tmpdir + ImgList[i]);

		//File.delete(tmpdir + ImgList[i]);

		// rewrite output in output dir to propper file format
		trg = output_dir + "transformed_" + ImgList[i];
		src = output_dir + "/result.tif";

		// remove trafo output if it already exists
		if (File.exists(trg)) {
			File.delete(trg);
		}
		File.rename(src, trg);
	}
	File.delete(trafo_file);
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

function save_custom(image, path, keep, output_type){
	selectWindow(image);

	if (is("Inverting LUT")) {
		run("Invert LUT");
	}

	if (output_type == "tif") {
		saveAs("tif", path + image);
	}

	if (output_type == "nrrd") {
		print(path + image);
		run("Nrrd ... ", "nrrd=" + path + image + ".nrrd");
	}
	
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

function Smooth(image, R, mag_lvl, mode){
	
	Ext.CLIJ2_clear();
	roiManager("reset");
	selectWindow(image);
	getDimensions(width, height, channels, slices, frames);
	getPixelSize(unit, pixelWidth, pixelHeight);
	
	// downsample to reg_width so that masks have same width
	f = reg_width/width;
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

	sigma = getWidth() / 175.0;
	
	selectWindow(name);
	Ext.CLIJ2_push(name);

	// subtract background
	Ext.CLIJ2_convertFloat(name, Float);
	Ext.CLIJ2_bottomHatSphere(Float, tmp, 10, 10, 1);
	Ext.CLIJ2_gaussianBlur2D(tmp, Float, sigma, sigma);

	// normalize
	Ext.CLIJ2_getMeanOfAllPixels(Float, mean);
	Ext.CLIJ2_standardDeviationOfAllPixels(Float);
	std = getResult("StandardDeviation", nResults-1);
	Ext.CLIJ2_addImageAndScalar(Float, tmp, (-1) * mean);
	Ext.CLIJ2_multiplyImageAndScalar(Float, tmp, (100.0/std));

	// return image
	close(name);
	Ext.CLIJ2_pull(tmp);
	name = getTitle();
	run("Set Scale...", "distance=1 known=" + pixelWidth/f  + " unit=" + unit);
	return name;
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
