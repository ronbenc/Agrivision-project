close all; clear all;

TRAIN_VAL_SPLIT = 'train';   %%% 'train'    'val'

STATS_FILE = 'stats.mat';
load(STATS_FILE);

if strcmp(TRAIN_VAL_SPLIT,'val')
    NUM_LABELS = NUM_LABELS(12902:end,:);
    FILE_NAMES = FILE_NAMES(12902:end,1);
    in_file = 'C:/alon/seg_test_1/pytorch-semseg-master/configs/agri2020_val_small.txt';
    scan_str = '%s%s';
    print_str = 'val';
    out_file = 'C:/alon/seg_test_1/pytorch-semseg-master/configs/agri2020_val_862.txt';
elseif strcmp(TRAIN_VAL_SPLIT,'train')
    NUM_LABELS = NUM_LABELS(1:12901,:);
    FILE_NAMES = FILE_NAMES(1:12901,1);
    in_file = 'C:/alon/seg_test_1/pytorch-semseg-master/configs/agri2020_train_ep_3888.txt';
    scan_str = '%s%s%s%s';
    print_str = 'train';
    out_file = 'C:/alon/seg_test_1/pytorch-semseg-master/configs/agri2020_train_3888.txt';
end

plant_pixels = NUM_LABELS(:,3:4);
num_plant_pixels = sum(NUM_LABELS(:,3:4),2);
expert_idx = find(num_plant_pixels > 0);
FILE_NAMES = FILE_NAMES(expert_idx);
f = fopen(in_file,'r');
F1 = textscan(f,scan_str);
fclose(f);
F1 = [F1{2};FILE_NAMES];
F1 = unique(F1);

f = fopen(out_file,'w');
for ii = 1:numel(F1)
    f_name = F1{ii};
    if strcmp(TRAIN_VAL_SPLIT,'val')
        fprintf(f,'%s  \t%s\n', print_str, f_name);
    elseif strcmp(TRAIN_VAL_SPLIT,'train')
        idx1 = ii;
        idx2 = ii;
        fprintf(f,'%s  \t%s \t%05d \t%05d\n', print_str, f_name, idx1, idx2 );
    end   
end
fclose(f);


