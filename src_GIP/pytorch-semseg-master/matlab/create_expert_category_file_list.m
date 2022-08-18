clear all; close all;

STATS_FILE = 'stats.mat';
load(STATS_FILE);

NUM_LABELS = NUM_LABELS(12902:end,:);
FILE_NAMES = FILE_NAMES(12902:end,:);

num_plant_pixels2 = sum(NUM_LABELS(:,3),2);
expert_idx2 = find(num_plant_pixels2 > 0);
num_plant_pixels6 = sum(NUM_LABELS(:,7),2);
expert_idx6 = find(num_plant_pixels6 > 0);
s = floor(numel(expert_idx6)/numel(expert_idx2));
expert_idx6 = expert_idx6(1:s:end);
expert_idx = unique([expert_idx2;expert_idx6]);
num_pics = numel(expert_idx);

file_list = 'C:\alon\seg_test_1\pytorch-semseg-master\configs\agri2020_val_ep26_892.txt';
f = fopen(file_list,'w');
for ii = 1:num_pics
    idx = expert_idx(ii);
    str1 = FILE_NAMES{idx,2};
    str1 = str1(1:end-1);
    str2 = FILE_NAMES{idx,1};
    fprintf(f,'%s  \t%s\n', str1, str2);
end
fclose(f);

NUM_LABELS = NUM_LABELS(1:12901,:);
FILE_NAMES = FILE_NAMES(1:12901,:);
plant_pixels = NUM_LABELS(:,3:4);
num_plant_pixels = sum(NUM_LABELS(:,3:4),2);
expert_idx = find(num_plant_pixels > 0);
num_pics = numel(expert_idx);
file_list = 'C:\alon\seg_test_1\pytorch-semseg-master\configs\agri2020_train_ep23_1911.txt';
f = fopen(file_list,'w');
for ii = 1:num_pics
    idx = expert_idx(ii);
    str1 = FILE_NAMES{idx,2};
    str1 = str1(1:end-1);
    str2 = FILE_NAMES{idx,1};
    fprintf(f,'%s  \t%s\n', str1, str2);
end
fclose(f);


% NUM_LABELS = NUM_LABELS(12902:end,:);
% FILE_NAMES = FILE_NAMES(12902:end,:);
% plant_pixels = NUM_LABELS(:,3);
% num_plant_pixels = sum(NUM_LABELS(:,3),2);
% expert_idx = find(num_plant_pixels > 0);
% num_pics = numel(expert_idx);
% file_list = 'C:\alon\seg_test_1\pytorch-semseg-master\configs\agri2020_val_cat2.txt';
% f = fopen(file_list,'w');
% for ii = 1:num_pics
%     idx = expert_idx(ii);
%     str1 = FILE_NAMES{idx,2};
%     str1 = str1(1:end-1);
%     str2 = FILE_NAMES{idx,1};
%     fprintf(f,'%s  \t%s\n', str1, str2);
% end
% fclose(f);