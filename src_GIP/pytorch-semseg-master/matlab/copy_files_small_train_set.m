clear all; close all;

% OUT_COLOR_DIR = 'C:\alon\datasets\Agriculture-Vision\train_small\gt_colors\';
% IN_RGB_DIR = 'C:\alon\datasets\Agriculture-Vision\train\images\rgb\';
% IN_NIR_DIR = 'C:\alon\datasets\Agriculture-Vision\train\images\nir\';
% %IN_NDVI_DIR = 'C:\alon\datasets\Agriculture-Vision\train\images\ndvi\';
% IN_LABEL_DIR = 'C:\alon\datasets\Agriculture-Vision\train\gt_labels\';
% OUT_IMG_DIR = 'C:\alon\datasets\Agriculture-Vision\train_small\images\';
% OUT_RGB_DIR = [OUT_IMG_DIR,'rgb\'];
% OUT_NIR_DIR = [OUT_IMG_DIR,'nir\'];
% %OUT_NDVI_DIR = [OUT_IMG_DIR,'ndvi\'];
% OUT_LABEL_DIR = 'C:\alon\datasets\Agriculture-Vision\train_small\gt_labels\';


% OUT_COLOR_DIR = 'C:\alon\datasets\Agriculture-Vision\val_small\gt_colors\';
% IN_RGB_DIR = 'C:\alon\datasets\Agriculture-Vision\val\images\rgb\';
% IN_NIR_DIR = 'C:\alon\datasets\Agriculture-Vision\val\images\nir\';
% IN_LABEL_DIR = 'C:\alon\datasets\Agriculture-Vision\val\gt_labels\';
% OUT_IMG_DIR = 'C:\alon\datasets\Agriculture-Vision\val_small\images\';
% OUT_RGB_DIR = [OUT_IMG_DIR,'rgb\'];
% OUT_NIR_DIR = [OUT_IMG_DIR,'nir\'];
% OUT_LABEL_DIR = 'C:\alon\datasets\Agriculture-Vision\val_small\gt_labels\';


% OUT_COLOR_DIR = 'C:\alon\datasets\Agriculture-Vision\zz_expert1_small\gt_colors\';
% IN_RGB_DIR = 'C:\alon\datasets\Agriculture-Vision\zz_expert1\images\rgb\';
% IN_NIR_DIR = 'C:\alon\datasets\Agriculture-Vision\zz_expert1\images\nir\';
% IN_LABEL_DIR = 'C:\alon\datasets\Agriculture-Vision\zz_expert1\gt_labels\';
% OUT_IMG_DIR = 'C:\alon\datasets\Agriculture-Vision\zz_expert1_small\images\';
% OUT_RGB_DIR = [OUT_IMG_DIR,'rgb\'];
% OUT_NIR_DIR = [OUT_IMG_DIR,'nir\'];
% OUT_LABEL_DIR = 'C:\alon\datasets\Agriculture-Vision\zz_expert1_small\gt_labels\';


OUT_COLOR_DIR = 'C:\alon\datasets\Agriculture-Vision\train_ep_3888\gt_colors\';
IN_RGB_DIR = 'C:\alon\datasets\Agriculture-Vision\train\images\rgb\';
IN_NIR_DIR = 'C:\alon\datasets\Agriculture-Vision\train\images\nir\';
IN_LABEL_DIR = 'C:\alon\datasets\Agriculture-Vision\train\gt_labels\';
OUT_IMG_DIR = 'C:\alon\datasets\Agriculture-Vision\train_ep_3888\images\';
OUT_RGB_DIR = [OUT_IMG_DIR,'rgb\'];
OUT_NIR_DIR = [OUT_IMG_DIR,'nir\'];
OUT_LABEL_DIR = 'C:\alon\datasets\Agriculture-Vision\train_ep_3888\gt_labels\';


if ~exist(OUT_IMG_DIR,'dir')
    mkdir(OUT_IMG_DIR)
end
if ~exist(OUT_RGB_DIR,'dir')
    mkdir(OUT_RGB_DIR)
end
if ~exist(OUT_NIR_DIR,'dir')
    mkdir(OUT_NIR_DIR)
end
% if ~exist(OUT_NDVI_DIR,'dir')
%     mkdir(OUT_NDVI_DIR)
% end
if ~exist(OUT_LABEL_DIR,'dir')
    mkdir(OUT_LABEL_DIR)
end

num_files = 0;
FILE_NAMES = [];
all_names = dir(OUT_COLOR_DIR);
for ii = 1:numel(all_names)
    curr_name = all_names(ii).name;
    if contains(curr_name,'gt_color.png')
        num_files = num_files + 1;
        FILE_NAMES{num_files,1} = all_names(ii).name; 
    end
end
%save('small_balanced_train_set','FILE_NAMES');

for ii = 1:numel(FILE_NAMES)
    disp(ii);
    curr_name = FILE_NAMES{ii};
    tmp = strsplit(curr_name,'_gt_color');
    prefix = tmp{1};
    
    curr_name = [prefix,'.jpg'];
    in_file = [IN_RGB_DIR,curr_name];
    out_file = [OUT_RGB_DIR,curr_name];
    copyfile(in_file,out_file);
    
    in_file = [IN_NIR_DIR,curr_name];
    out_file = [OUT_NIR_DIR,curr_name];
    copyfile(in_file,out_file);
    
%     curr_name = [prefix,'.png'];
%     in_file = [IN_NDVI_DIR,curr_name];
%     out_file = [OUT_NDVI_DIR,curr_name];
%     copyfile(in_file,out_file);
    
    curr_name = [prefix,'_gt_label.png'];
    in_file = [IN_LABEL_DIR,curr_name];
    out_file = [OUT_LABEL_DIR,curr_name];
    copyfile(in_file,out_file);
    
end

