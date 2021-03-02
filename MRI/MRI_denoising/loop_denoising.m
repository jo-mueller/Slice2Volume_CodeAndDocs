root = 'E:\Promotion\Projects\2020_Slice2Volume\Data';

mice = {'P2A_C3H_M10'};

for m = 1:length(mice)
    MRI_path = fullfile(root, mice{m}, 'MRI');
    
    timepoints = dir(MRI_path);
    for t = 1:length(timepoints)
        
        if strcmp(timepoints(t).name, '.')
            continue;
        end
        
        if strcmp(timepoints(t).name, '..')
            continue;
        end
        
        % iterate over timepoints
        tp_path = fullfile(MRI_path, timepoints(t).name);
        img_dirs = dir(tp_path);
        
        for i = 1:length(img_dirs)
            
            % skip all non-denoised directories
           if length(strfind(img_dirs(i).name, 'denoised')) == 0
               continue;
           end
           
           % find sequence data
           denoised_dir = fullfile(tp_path, img_dirs(i).name);
           sequences = dir(denoised_dir);
           
           for s = 1:length(sequences)
               
               if strcmp(sequences(s).name, '.')
                    continue;
               end
               if strcmp(sequences(s).name, '..')
                   continue;
               end
               
               seq = sequences(s).name;
               if ~strcmp(seq(end-2:end), 'nii')
                   continue;
               end
               if strcmp(seq(1:2), 'T1') || strcmp(seq(1:2), 'T2')
                   
                   seq = fullfile(denoised_dir, seq);
                   disp(['Denoising: ', seq]);
                   denoise_MRI(seq);
                   
               end
               
           end
        end
    end
end