clear all; close all;

BASE_DIR = 'C:\alon\datasets\Agriculture-Vision\';
STATS_FILE = 'stats.mat';

OUT_DIR = [BASE_DIR,'zz_expert1\'];
%OUT_DIR = [BASE_DIR,'zz_expert1_val_only\'];
%OUT_DIR = [BASE_DIR,'zz_expert1_train_only\'];
OUT_IMG_DIR = [OUT_DIR,'images\'];
OUT_RGB_DIR = [OUT_IMG_DIR,'rgb\'];
OUT_NIR_DIR = [OUT_IMG_DIR,'nir\'];
OUT_LABEL_DIR = [OUT_DIR,'gt_labels\'];
OUT_COLOR_DIR = [OUT_DIR,'gt_colors\'];

if ~exist(OUT_DIR,'dir')
    mkdir(OUT_DIR)
end
if ~exist(OUT_IMG_DIR,'dir')
    mkdir(OUT_IMG_DIR)
end
if ~exist(OUT_RGB_DIR,'dir')
    mkdir(OUT_RGB_DIR)
end
if ~exist(OUT_NIR_DIR,'dir')
    mkdir(OUT_NIR_DIR)
end
if ~exist(OUT_LABEL_DIR,'dir')
    mkdir(OUT_LABEL_DIR)
end
if ~exist(OUT_COLOR_DIR,'dir')
    mkdir(OUT_COLOR_DIR)
end



load(STATS_FILE);
plant_pixels = NUM_LABELS(:,3:4);
num_plant_pixels = sum(NUM_LABELS(:,3:4),2);
expert_idx = find(num_plant_pixels > 0);


num_pics = numel(expert_idx);
for ii = 1:num_pics
    disp(ii);
    idx = expert_idx(ii);
    case_dir = FILE_NAMES{idx,2};
    prefix = FILE_NAMES{idx,1};
    
    if strcmp(case_dir,'val\')
        continue
    end    
    
    curr_name = [prefix,'.jpg'];
    in_file = [BASE_DIR,case_dir,'images\rgb\',curr_name];
    out_file = [OUT_RGB_DIR,curr_name];
    copyfile(in_file,out_file);
    
    curr_name = [prefix,'.jpg'];
    in_file = [BASE_DIR,case_dir,'images\nir\',curr_name];
    out_file = [OUT_NIR_DIR,curr_name];
    copyfile(in_file,out_file);
    
    curr_name = [prefix,'_gt_label.png'];
    in_file = [BASE_DIR,case_dir,'gt_labels\',curr_name];
    out_file = [OUT_LABEL_DIR,curr_name];
    copyfile(in_file,out_file);
    
    curr_name = [prefix,'_gt_color.png'];
    in_file = [BASE_DIR,case_dir,'gt_colors\',curr_name];
    out_file = [OUT_COLOR_DIR,curr_name];
    copyfile(in_file,out_file);
end
