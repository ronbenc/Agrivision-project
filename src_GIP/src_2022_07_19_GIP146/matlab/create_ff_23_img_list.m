clear all; close all;

BASE_DIR = 'C:\alon\datasets\Agriculture-Vision\train\full_field';
out_file = 'train_full_field_zz.txt';  %'train_full_field.txt'
case_dir = 'train';

SKIP = 64;
SHIFT = 3;
MIN_VALID_RATIO = 0.5;
IMG_SIZE = 512;
OOB_CATEGORY = 7;
MIN_VALID = MIN_VALID_RATIO * IMG_SIZE * IMG_SIZE;
MIN_VALID_23 = 10;
do_visualize = 0;
do_write_images = 1;
OUT_DIR_VIS = 'C:\alon\seg_test_1\zz_tmp_pics\';
OUT_DIR_RGB = 'C:\alon\datasets\Agriculture-Vision\train_ff_23\rgb\';
OUT_DIR_NIR = 'C:\alon\datasets\Agriculture-Vision\train_ff_23\nir\';
OUT_DIR_LBL = 'C:\alon\datasets\Agriculture-Vision\train_ff_23\gt_labels\';
if do_write_images == 1
    if ~exist(OUT_DIR_RGB,'dir')
        mkdir(OUT_DIR_RGB);
    end
    if ~exist(OUT_DIR_NIR,'dir')
        mkdir(OUT_DIR_NIR);
    end
    if ~exist(OUT_DIR_LBL,'dir')
        mkdir(OUT_DIR_LBL);
    end
end

all_files = dir(BASE_DIR);
label_files = cell(0);
num_fields = 0;
for ii = 1:numel(all_files)
    curr_name = all_files(ii).name;
    if contains(curr_name,'_gt_label.png')
        num_fields = num_fields + 1;
        label_files{num_fields,1} = curr_name;
    end
end

num_pics = zeros(num_fields,1);
f = fopen(out_file,'w');
for n = 1:num_fields  %1:num_fields   7 45 
    disp(n);
    prefix = label_files{n};
    prefix = strsplit(prefix,'_');
    prefix = prefix{1};
    lbl_file = [BASE_DIR,'\',label_files{n}];
    lbl = imread(lbl_file);
    if do_visualize == 1 || do_write_images == 1 
        rgb_file = [BASE_DIR,'\',prefix,'_rgb.png'];
        nir_file = [BASE_DIR,'\',prefix,'_nir.png'];    
        rgb = imread(rgb_file);
        nir = imread(nir_file);
    end
    
    [h,w] = size(lbl);
    
    valid_idx_23 = lbl(lbl == 2 | lbl == 3);
    num_valid_23 = numel(valid_idx_23);
    if num_valid_23 < MIN_VALID_23
        continue
    end
    
    for ii = 1+SHIFT : SKIP : h-IMG_SIZE+1
        for jj = 1+SHIFT : SKIP : w-IMG_SIZE+1
            curr_lbl = lbl(ii:ii+IMG_SIZE-1,jj:jj+IMG_SIZE-1);
            valid_idx = curr_lbl(curr_lbl < OOB_CATEGORY);
            num_valid = numel(valid_idx);
            valid_idx_23 = curr_lbl(curr_lbl == 2 | curr_lbl == 3);
            num_valid_23 = numel(valid_idx_23);
            
            if num_valid >= MIN_VALID && num_valid_23 >= MIN_VALID_23
                num_pics(n) = num_pics(n) + 1;
                u0 = ii - 1;
                v0 = jj - 1;
                %fprintf(f,'%s\t%s\t%d\t%d\n',case_dir,prefix,u0,v0);
                file_str = [prefix,'_',int2str(u0),'_',int2str(v0)];
                fprintf(f,'%s\t%s\t%d\t%d\n',case_dir,file_str,u0,v0);
                
                if do_visualize == 1
                    curr_rgb = rgb(ii:ii+IMG_SIZE-1,jj:jj+IMG_SIZE-1,:);
                    out_rgb_file = [OUT_DIR_VIS,prefix,'_',int2str(ii),'_',int2str(jj),'.png'];
                    imwrite(curr_rgb,out_rgb_file);
                end
                if do_write_images == 1 
                    curr_rgb = rgb(ii:ii+IMG_SIZE-1,jj:jj+IMG_SIZE-1,:);
                    out_rgb_file = [OUT_DIR_RGB,file_str,'.png'];
                    imwrite(curr_rgb,out_rgb_file);
                    curr_nir = nir(ii:ii+IMG_SIZE-1,jj:jj+IMG_SIZE-1);
                    out_nir_file = [OUT_DIR_NIR,file_str,'.png'];
                    imwrite(curr_nir,out_nir_file);
                    
                    out_lbl_file = [OUT_DIR_LBL,file_str,'.png'];
                    imwrite(curr_lbl,out_lbl_file);
                end
            end            
        end
    end
    dummy = 0;
end
fclose(f);

