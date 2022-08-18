clear all; close all;
BASE_DIR = 'C:\alon\datasets\Agriculture-Vision\';

LABEL_NAMES = { 'boundaries\' ...
                'masks\', ...
                'labels\cloud_shadow\', ...
                'labels\double_plant\', ...
                'labels\planter_skip\', ...
                'labels\standing_water\', ...
                'labels\waterway\', ...
                'labels\weed_cluster\'};

IMG_SIZE = [512,512];
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

N_LABELS = zeros(num_files,8);
for ii = 1:num_files
    disp(ii)
    case_dir = FILE_NAMES{ii,2};
    prefix = FILE_NAMES{ii,1};
    
    I = uint8(zeros(IMG_SIZE(1),IMG_SIZE(2),8));
    for jj = 1:8
        I(:,:,jj) = imread([BASE_DIR,case_dir,LABEL_NAMES{jj},prefix,'.png']);        
    end
    I = logical(I);
    out_of_bounds = ~(I(:,:,1) & I(:,:,2));
    N_LABELS(ii,8) = numel(find(out_of_bounds));
    for jj = 3:8
        N_LABELS(ii,jj-1) = numel(find(I(:,:,jj)));
    end
%     background = ~out_of_bounds;
%     for jj = 3:8
%         background = background & ~I(:,:,jj);
%     end
    background = ~I(:,:,3);
    for jj = 4:8
        background = background & ~I(:,:,jj);
    end
    N_LABELS(ii,1) = numel(find(background));    
end

RES_FILE = 'stats_num_labels';
save(RES_FILE,'FILE_NAMES','N_LABELS','BASE_DIR','LABEL_NAMES','IMG_SIZE');

%%% images with multi-prediction-pixels
clear all;
load('stats_num_labels_2');
MIN_OVERLAPPING_PIXELS = 0;
N_CATEGORIES = 6;
A = sum(N_LABELS(:,1:7),2);
A = A - IMG_SIZE(1)*IMG_SIZE(2);
idx_multi = find(A>MIN_OVERLAPPING_PIXELS);
num_multi = numel(idx_multi);
num_overlap = A(idx_multi);
num_overlap_check = zeros(num_multi,1);
overlapping_labels = zeros(N_CATEGORIES,N_CATEGORIES,num_multi);
multi_labels = cell(num_multi,1);
for ii = 1:num_multi
    idx = idx_multi(ii);
    disp([ii,idx]);
    case_dir = FILE_NAMES{idx,2};
    prefix = FILE_NAMES{idx,1};
    I = uint8(zeros(IMG_SIZE(1),IMG_SIZE(2),N_CATEGORIES));
    for jj = 1:N_CATEGORIES
        I(:,:,jj) = imread([BASE_DIR,case_dir,LABEL_NAMES{jj+2},prefix,'.png']);        
    end
    I = logical(I);
    for jj = 1:N_CATEGORIES-1
        for kk = jj+1:N_CATEGORIES
            I_overlap = I(:,:,jj) & I(:,:,kk); 
            tmp_overlap = numel(find(I_overlap));
            overlapping_labels(jj,kk,ii) = tmp_overlap;
            overlapping_labels(kk,jj,ii) = tmp_overlap;  
            if tmp_overlap > 0
                multi_labels{ii} = [multi_labels{ii},[jj,kk]];
            end
        end
    end 
    curr_overlap = overlapping_labels(:,:,ii);
    num_overlap_check(ii) = sum(curr_overlap(:))/2;
end
RES_FILE = 'stats_overlapping_labels';
save(RES_FILE,'FILE_NAMES','overlapping_labels','multi_labels','num_overlap','idx_multi');
