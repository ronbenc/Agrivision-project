clear all; close all;

ALL_FILES_LIST = 'C:\alon\seg_test_1\pytorch-semseg-master\configs\agri2020_all.txt';

f = fopen('C:\alon\seg_test_1\pytorch-semseg-master\configs\agri2020_train_all.txt','r');
F1 = textscan(f,'%s%s%s%s');
fclose(f);
FILE_NAMES = [F1{1},F1{2}];

f = fopen('C:\alon\seg_test_1\pytorch-semseg-master\configs\agri2020_val_all.txt','r');
F1 = textscan(f,'%s%s');
fclose(f);
FILE_NAMES = [FILE_NAMES;[F1{1},F1{2}]];

f = fopen('C:\alon\seg_test_1\pytorch-semseg-master\configs\agri2020_test_all.txt','r');
F1 = textscan(f,'%s%s');
fclose(f);
FILE_NAMES = [FILE_NAMES;[F1{1},F1{2}]];

save('file_names','FILE_NAMES');