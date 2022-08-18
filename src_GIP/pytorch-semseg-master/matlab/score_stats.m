clear all; close all;

score_file = 'C:\alon\seg_test_1\pytorch-semseg-master\runs\tst_rdse\2021_09_07_13_19_09_agri6_rdse_agct_3a_me1\out_t_all\img_stats.txt';

NUM_CATEGORIES = 6; %%% not including background
MIN_SCORE = 0.7;    % 0.6
MIN_PICS_PER_FIELD = 4;

T = readtable(score_file);
file_names = T{:,15};
num_files = numel(file_names);
prefixes = cell(num_files,1);
for ii = 1:num_files
    tmp = strsplit(file_names{ii},'_');
    prefixes{ii} = tmp{1};
end
unique_prefixes = unique(prefixes);
num_unique_prefixes = numel(unique_prefixes);

ok_list = [];   %cell(0);
num_ok = 0;
for ii = 1:num_unique_prefixes
    idx = find(strcmp(prefixes,unique_prefixes{ii}) == 1);
    num_idx = numel(idx);
    for jj = 1:NUM_CATEGORIES
        col_idx = 2*(jj+1);
        curr_scores = T{idx,col_idx};
        
        for kk = 1:num_idx
            if curr_scores(kk) > MIN_SCORE
                num_ok = num_ok + 1;
                %ok_list{num_ok,1} = file_names{idx(kk)};
                ok_list = [ok_list;idx(kk)];
            end            
        end       
        
    end    
end

ok_list = unique(ok_list);
ok_files = file_names(ok_list);

%%% merge with categories 2,3
STATS_FILE = 'stats.mat';
load(STATS_FILE);
NUM_LABELS = NUM_LABELS(1:12901,:);
FILE_NAMES = FILE_NAMES(1:12901,1);
num_plant_pixels = sum(NUM_LABELS(:,3:4),2);
expert_idx = find(num_plant_pixels > 0);
FILE_NAMES = FILE_NAMES(expert_idx);
F1 = [ok_files;FILE_NAMES];
F1 = unique(F1);

out_file = 'C:/alon/seg_test_1/pytorch-semseg-master/configs/agri2020_train_xxx4999.txt';
print_str = 'train';
f = fopen(out_file,'w');
for ii = 1:numel(F1)
    f_name = F1{ii};
    idx1 = ii;
    idx2 = ii;
    fprintf(f,'%s  \t%s \t%05d \t%05d\n', print_str, f_name, idx1, idx2 );   
end
fclose(f);
