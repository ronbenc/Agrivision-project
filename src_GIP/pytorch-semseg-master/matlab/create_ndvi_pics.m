clear all; close all;

BASE_DIR = 'C:\alon\datasets\Agriculture-Vision\';
IMG_SIZE = [512,512];
CASE_DIR = 'train\';  %%% 'val\'  'train\'    'test\'
RGB_DIR = [BASE_DIR,CASE_DIR,'images\rgb\'];
NIR_DIR = [BASE_DIR,CASE_DIR,'images\nir\'];
NDVI_DIR = [BASE_DIR,CASE_DIR,'images\ndvi\'];

if ~exist(NDVI_DIR,'dir')
    mkdir(NDVI_DIR);
end

num_files = 0;
FILE_NAMES = [];
all_names = dir(RGB_DIR);
for ii = 1:numel(all_names)
    curr_name = all_names(ii).name;
    if contains(curr_name,'.jpg')
        num_files = num_files + 1;
        tmp = strsplit(curr_name,'.jpg');
        prefix = tmp{1};
        FILE_NAMES{num_files,1} = prefix; 
    end
end

min_max_vals = nan(num_files,2);

for ii = 1:num_files 
   disp(ii);
   RGB = imread([RGB_DIR,FILE_NAMES{ii},'.jpg']); 
   NIR = imread([NIR_DIR,FILE_NAMES{ii},'.jpg']); 
   R = double(RGB(:,:,1));
   NIR = double(NIR);   
   NDVI = (NIR - R)./(NIR + R);
   NDVI = uint8(127.5 * (NDVI + 1));   
   out_file = [NDVI_DIR,FILE_NAMES{ii},'.png']; 
   imwrite(NDVI,out_file);
   min_max_vals(ii,1) = min(NDVI(:));
   min_max_vals(ii,2) = max(NDVI(:));
end

figure;plot(1:num_files,min_max_vals(:,1),'.b',1:num_files,min_max_vals(:,2),'.r'); 


