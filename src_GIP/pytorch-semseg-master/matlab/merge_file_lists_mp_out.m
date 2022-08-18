clear all; close all;

str_multi = 'multi';

% in_file = 'C:/alon/seg_test_1/pytorch-semseg-master/configs/agri2020_val_862_mp.txt';
% out_file = 'C:/alon/seg_test_1/pytorch-semseg-master/configs/agri2020_val_862_mp0.txt';
% in_file = 'C:/alon/seg_test_1/pytorch-semseg-master/configs/agri2020_val_all_mp.txt';
% out_file = 'C:/alon/seg_test_1/pytorch-semseg-master/configs/agri2020_val_all_mp0.txt';
% TRAIN_VAL_SPLIT = 'val';

% in_file = 'C:/alon/seg_test_1/pytorch-semseg-master/configs/agri2020_train_3888_mp.txt';
% out_file = 'C:/alon/seg_test_1/pytorch-semseg-master/configs/agri2020_train_3888_mp0.txt';
in_file = 'C:/alon/seg_test_1/pytorch-semseg-master/configs/agri2020_train_all_mp.txt';
out_file = 'C:/alon/seg_test_1/pytorch-semseg-master/configs/agri2020_train_all_mp0.txt';
TRAIN_VAL_SPLIT = 'train';

T1 = readtable(in_file);
T1 = [T1.Properties.VariableDescriptions;T1];
files_1 = T1{:,2};
num_1 = numel(files_1);

file_list = cell(0);
num_files = 0;
for ii = 1:num_1
    is_ok = 1;
    curr_file_1 = files_1{ii};
    if contains(curr_file_1,str_multi)
        continue
    end
    for jj = 1:num_1
        if ii == jj
            continue
        end
        curr_file_2 = files_1{jj};
        if contains(curr_file_2,curr_file_1) && contains(curr_file_2,str_multi)
            is_ok = 0;
            continue    
        end
        if contains(curr_file_1,curr_file_2) && contains(curr_file_1,str_multi)
            is_ok = 0;
            continue                      
        end
    end  
    if is_ok == 1
        num_files = num_files + 1;
        file_list{num_files,1} = curr_file_1;  
    end
end

f = fopen(out_file,'w');
for ii = 1:num_files
    f_name = file_list{ii};
    if strcmp(TRAIN_VAL_SPLIT,'val')
        fprintf(f,'%s  \t%s\n', TRAIN_VAL_SPLIT, f_name);
    elseif strcmp(TRAIN_VAL_SPLIT,'train')
        idx1 = ii;
        idx2 = ii;
        fprintf(f,'%s  \t%s \t%05d \t%05d\n', TRAIN_VAL_SPLIT, f_name, idx1, idx2 );
    end   
end
fclose(f);

