clear all; close all;

BASE_DIR = 'C:\alon\datasets\Agriculture-Vision\';
IMG_SIZE = [512,512];
CASE_DIR = 'train\';  %%% 'val\'  'train\'    'test\'
RGB_DIR = [BASE_DIR,CASE_DIR,'images\rgb\'];
NIR_DIR = [BASE_DIR,CASE_DIR,'images\nir\'];
BOUNDARY_DIR = [BASE_DIR,CASE_DIR,'boundaries\'];
MASK_DIR = [BASE_DIR,CASE_DIR,'masks\'];
LABEL_NAMES = {'background', ...
                'cloud_shadow', ...
                'double_plant', ...
                'planter_skip', ...
                'standing_water', ...
                'waterway', ...
                'weed_cluster'};
%LABEL_ORDER = [1,2,3,4,5,6,7];
LABEL_ORDER = [1,2,7,6,5,4,3];

label_colors =  [128, 128, 128; ... %%% 
                   0,   0, 255; ...
                   0, 255,   0; ...
                   0, 255, 255; ...
                 255,   0,   0; ...
                 255,   0, 255; ...
                 255, 255,   0; ...
                   0,   0,   0];    %%% out-of-bounds
             
num_labels = numel(LABEL_NAMES);
LABEL_DIRS = cell(num_labels,1);
for ii = 1:num_labels
    LABEL_DIRS{ii,1} = [BASE_DIR,CASE_DIR,'labels\',LABEL_NAMES{ii},'\'];
end

OUT_DIR = [BASE_DIR,CASE_DIR,'full_field_2\'];
if ~exist(OUT_DIR,'dir')
    mkdir(OUT_DIR);
end

num_files = 0;
FILE_NAMES = [];
all_names = dir(RGB_DIR);
for ii = 1:numel(all_names)
    curr_name = all_names(ii).name;
    if contains(curr_name,'.jpg') && ~contains(curr_name,'_multi_')
        num_files = num_files + 1;
        FILE_NAMES{num_files,1} = all_names(ii).name;
    end
end
NAME_INFO = cell(num_files,2);
COORDS = zeros(num_files,4);
for ii = 1:num_files
    curr_name = FILE_NAMES{ii};    
    curr_name = strsplit(curr_name,'.');
    curr_name = curr_name{1};
    NAME_INFO{ii,1} = curr_name;
    tmp = strsplit(curr_name,'_');
    field_id = tmp{1};
    NAME_INFO{ii,2} = field_id;
    curr_coords = strsplit(tmp{2},'-');      
    for jj = 1:4
        COORDS(ii,jj) = str2num(curr_coords{jj});
    end
end

agri_field_ids = unique(NAME_INFO(:,2));
num_fields = numel(agri_field_ids);
label_diff = nan(num_fields,1);
for ii = 1:num_fields
    prefix = agri_field_ids{ii};    
    img_idx = find(strcmp(NAME_INFO(:,2),prefix));
    num_images = numel(img_idx);
    minH = min(COORDS(img_idx,2));
    maxH = max(COORDS(img_idx,4));
    minW = min(COORDS(img_idx,1));
    maxW = max(COORDS(img_idx,3));
    
    h = maxH - minH;
    w = maxW - minW; 
    full_rgb = uint8(zeros(h,w,3));
    full_nir = uint8(zeros(h,w));
    if ~strcmp(CASE_DIR,'test\')
        full_gt_label = uint8(num_labels*ones(h,w));
        full_gt_color = uint8(zeros(h,w,3));
    end
    
    for jj = 1:num_images
        curr_id = img_idx(jj);    
        curr_prefix = NAME_INFO{curr_id,1};
        %disp(curr_prefix);
        h0 = COORDS(curr_id,2) - minH + 1;
        h1 = COORDS(curr_id,4) - minH;
        w0 = COORDS(curr_id,1) - minW + 1;
        w1 = COORDS(curr_id,3) - minW;
        rgb = imread([RGB_DIR,curr_prefix,'.jpg']);
        nir = imread([NIR_DIR,curr_prefix,'.jpg']);
        full_rgb(h0:h1,w0:w1,:) = rgb;
        full_nir(h0:h1,w0:w1) = nir;
        
        if ~strcmp(CASE_DIR,'test\')
            boundary_file_name = [BOUNDARY_DIR,curr_prefix,'.png'];
            boundary = imread(boundary_file_name);
            mask_file_name = [MASK_DIR,curr_prefix,'.png'];
            mask = imread(mask_file_name);
            out_of_bounds = logical(boundary == 0 | mask == 0);
            background = true(IMG_SIZE(1),IMG_SIZE(2));            
            gt_label = uint8(zeros(IMG_SIZE(1),IMG_SIZE(2)));     
            
            for kk = 2:num_labels
                curr_label_ord = LABEL_ORDER(kk);
                label_file_name = [LABEL_DIRS{curr_label_ord,1},curr_prefix,'.png'];
                curr_label = imread(label_file_name);
                background = background & (curr_label==0);
                idx = find(curr_label == 255);
                gt_label(idx) = curr_label_ord-1;
            end    
            out_of_bounds = out_of_bounds & background;
            gt_label(out_of_bounds) = num_labels;  

            r = uint8(zeros(IMG_SIZE(1),IMG_SIZE(2)));
            g = uint8(zeros(IMG_SIZE(1),IMG_SIZE(2)));
            b = uint8(zeros(IMG_SIZE(1),IMG_SIZE(2)));
            gt_color = uint8(zeros(IMG_SIZE(1),IMG_SIZE(2),3));
            for kk = 1:num_labels+1
                idx = find(gt_label == (kk-1));
                r(idx) = label_colors(kk,1);
                g(idx) = label_colors(kk,2);
                b(idx) = label_colors(kk,3);
            end
            gt_color(:,:,1) = r;
            gt_color(:,:,2) = g;
            gt_color(:,:,3) = b;

            full_gt_label(h0:h1,w0:w1) = gt_label;
            full_gt_color(h0:h1,w0:w1,:) = gt_color;    
        end
    end    
    
    out_file = [OUT_DIR,prefix,'_rgb.png'];
    imwrite(full_rgb,out_file);
    out_file = [OUT_DIR,prefix,'_nir.png'];
    imwrite(full_nir,out_file);
    if ~strcmp(CASE_DIR,'test\')
        out_file = [OUT_DIR,prefix,'_gt_label.png'];
        imwrite(full_gt_label,out_file);
        out_file = [OUT_DIR,prefix,'_gt_color.png'];
        imwrite(full_gt_color,out_file);
    end
    
    %%% compare with prev
    prev_label_file = ['C:\alon\datasets\Agriculture-Vision\train\full_field\',prefix,'_gt_label.png'];
    full_prev = imread(prev_label_file);
    diff = int8(full_gt_label) - int8(full_prev);
    num_diff = numel(find(diff ~= 0));
    label_diff(ii,1) = num_diff;
    str = [int2str(ii),' : ',prefix,' : ',int2str(num_diff)];
    disp(str);
    d=0;
end

