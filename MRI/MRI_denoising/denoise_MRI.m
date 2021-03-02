    function []=denoise_MRI(input_file)


%% get MRI data in nifit format
[MRIpath, MRI_filename, ext] = fileparts(input_file);
MRI_filename = [MRI_filename, ext];
%[MRI_filename,MRIpath]=uigetfile('*.nii;*.nii.gz','select Nifti file for denoising');


% MRI_filename='Bl6 M8 w13_T2_FSE_3D_Centric_Partial_iso-small_6min_20191111142840_3.nii.gz';
% MRI_filename='Bl6 M8 w13_T1_GRE_SP_3D_iso_11min_20191111142840_5.nii.gz';
% MRI_filename='Bl6 M8 w13_MRI_Static_20191111142840_2.nii.gz';
% MRIpath='C:\ARBEIT\Mäuse\Bl6 M8 w13';

% MRI_filename='C3H M6 w7_T2_FSE_3D_Centric_Partial_iso-small_6min_20190425144354_2.nii.gz';
% MRI_filename='C3H M6 w7_T1_GRE_SP_3D_iso_11min_20190425144354_4.nii.gz';
% MRIpath='C:\ARBEIT\Mäuse\C3H M6 w7';


basename=strtok(MRI_filename,'.');


MRI=load_untouch_nii([MRIpath,filesep,MRI_filename]);

MRI.img=double(MRI.img);

M=max(MRI.img(:));
MRI.img=MRI.img./M;

%% select noise region
slice=round(size(MRI.img,3)/2);
test_slice=double(MRI.img(:,:,slice));

%figure;imagesc(test_slice);axis image;colormap(gray);axis off
%disp('---------------------------------------------------------')
%disp('##: create generous ROI around the image structure...')
%disp('##: ...double click when finished...')
%h=impoly;
%pos=wait(h);

margin = 10;
sz = size(test_slice);


 xi = [margin, sz(1) - margin, sz(1) - margin, margin];
 yi = [margin, margin, sz(1) - margin, sz(1) - margin];
noise_mask = poly2mask(xi,yi,sz(1), sz(1));
disp('##: OK...you stop clicking now')
disp('##: ...doing stuff...')
%noise_mask=h.createMask;
noise_mask=repmat(noise_mask,[1,1,size(MRI.img,3)]);
%% estimate sigma in noise region
noise_MRI=MRI.img;
noise_MRI(noise_mask==1)=[];

%figure;hist(noise_MRI,300);title('--> rician noise distribution')

sigma=std(noise_MRI);
disp(['##: sigma = ',num2str(sigma)])
drawnow
%% denoising BM3D: http://www.cs.tut.fi/~foi/GCF-BM3D/ 
MRI_filtered1=zeros(size(MRI.img));
MRI_filtered2=zeros(size(MRI.img));
for s=1:size(MRI.img,3)
    MRI_slice=double(MRI.img(:,:,s));
    %[~, MRI_filtered1(:,:,s)] = BM3D(1,MRI_slice,sigma*255);%x1 sigma
    [~, MRI_filtered2(:,:,s)] = BM3D(1,MRI_slice,sigma*255*1.5);%x1.5 sigma
end

MRI.img=(MRI_filtered1.*M);
save_untouch_nii(MRI,[MRIpath,filesep,basename,'_BM3D_1xsigma.nii.gz'])
MRI.img=(MRI_filtered2.*M);
save_untouch_nii(MRI,[MRIpath,filesep,basename,'_BM3D_15xsigma.nii.gz'])
%% anisotroic diffusion filtering
% MRI_filtered=zeros(size(MRI.img));
% for s=1:size(MRI.img,3)
%     MRI_slice=double(MRI.img(:,:,s));
%     MRI_filtered(:,:,s) = imdiffusefilt(MRI_slice);
% end
% 
% MRI.img=(MRI_filtered.*M);
% save_untouch_nii(MRI,[MRIpath,filesep,'Bl6M8w13_T2_FSE_3D_mat2019anisodiff.nii.gz'])
% % Idiffusion = imdiffusefilt(I);







%% display
% figure
% montage([test_slice y_est_sigma])
% 
% figure;
% subplot(2,1,1)
% imagesc(test_slice);axis image;colormap(gray);axis off
% subplot(2,1,2)
% imagesc(y_est);axis image;colormap(gray);axis off







disp('##: ...done.')
disp('---------------------------------------------------------')
    end