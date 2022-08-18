clear all; close all;

BASE_DIR = 'C:\alon\datasets\Agriculture-Vision\';
%CASE_DIR = 'train\';  %%% 'val\'  'train\'    'test\'
%RGB_DIR = [BASE_DIR,CASE_DIR,'images\rgb\'];
%LABEL_DIR = [BASE_DIR,CASE_DIR,'gt_labels\'];
RES_FILE = 'stats';
NUM_CATEGORIES = 8;

num_files = 0;
FILE_NAMES = [];
all_names = dir([BASE_DIR,'train\','gt_labels\']);
for ii = 1:numel(all_names)
    curr_name = all_names(ii).name;
    if contains(curr_name,'.png')
        num_files = num_files + 1;
        tmp = strsplit(curr_name,'_gt_label.png');
        prefix = tmp{1};
        FILE_NAMES{num_files,1} = prefix; 
        FILE_NAMES{num_files,2} = 'train\'; 
    end
end
all_names = dir([BASE_DIR,'val\','gt_labels\']);
for ii = 1:numel(all_names)
    curr_name = all_names(ii).name;
    if contains(curr_name,'.png')
        num_files = num_files + 1;
        tmp = strsplit(curr_name,'_gt_label.png');
        prefix = tmp{1};
        FILE_NAMES{num_files,1} = prefix; 
        FILE_NAMES{num_files,2} = 'val\'; 
    end
end

NUM_LABELS = zeros(num_files,NUM_CATEGORIES); 
for ii = 1:num_files
    I = imread([BASE_DIR,FILE_NAMES{ii,2},'gt_labels\',FILE_NAMES{ii,1},'gt_label.png']);
    disp(ii);
    for jj = 1:NUM_CATEGORIES
        idx = jj -1;
        numP = numel(I(I==idx));
        NUM_LABELS(ii,jj) = numP;
    end
end

save(RES_FILE,'FILE_NAMES','NUM_LABELS');

