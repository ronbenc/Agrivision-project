clear all; close all;

IN_GT_DIR = 'C:\alon\datasets\Agriculture-Vision\train_small\gt_colors\';
IN_RGB_DIR = 'C:\alon\datasets\Agriculture-Vision\train_small\images\rgb\';
OUT_DIR = 'C:\alon\datasets\Agriculture-Vision\train_small\tst1\';

num_files = 0;
FILE_NAMES = [];
all_names = dir(IN_GT_DIR);
for ii = 1:numel(all_names)
    curr_name = all_names(ii).name;
    if contains(curr_name,'gt_color.png')
        num_files = num_files + 1;
        tmp = strsplit(curr_name,'_gt_color');
        prefix = tmp{1};
        FILE_NAMES{num_files,1} = prefix; 
    end
end

for ii = 1:num_files   
   I1 = imread([IN_RGB_DIR,FILE_NAMES{ii},'.jpg']); 
   I2 = imread([IN_GT_DIR,FILE_NAMES{ii},'_gt_color.png']); 
   I = [I1,I2];
   I = imresize(I,0.5);
   out_file = [OUT_DIR,FILE_NAMES{ii},'_rg_gt.png']; 
   imwrite(I,out_file);
end