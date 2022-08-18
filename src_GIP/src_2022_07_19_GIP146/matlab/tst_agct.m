clear all; close all;

alpha_coeffs = [-1.0, 0.0, 0.0, 1.0, 0.0,    1.0, 0.0, 0.0, 1.0, 0.0; ...    % ~NDVI [RGBN0-nomin RGBN0-denom]
                -2.0, 0.0, 0.0, 2.0, 0.0,   1.0, 0.0, 0.0, 1.0, 1.0; ... % ~SAVI, note multiply by (L+1) 
                -2.5, 0.0, 0.0, 2.5, 0.0,   6.0, 0.0, 7.5, 1.0, 1.0; ...  % ~EVI
                
                0.0, -1.0, 0.0, 1.0, 0.0,   0.0, 1.0, 0.0, 0.0, 0.0; ...  % ~GCI 
                0.0, 0.0, 0.0, 1.0, 0.0,   0.0, 1.0, 0.0, 0.0, -1.0; ...  % ~GCI fake
                0.0, 0.0, -1.0, 1.0, 0.0,   -1.0, 0.0, 0.0, 1.0, 0.0; ... % ~SIPI !!![0.0, 0.0, -1.0, 1.0, 0.0,   -1.0, 0.0, 0.0, 1.0, 0.0] note fake N+R in denom # note +alpha[9] in denom                                          
                0.0, 0.0, -1.0, 1.0, 0.0,   1.0, 0.0, 0.0, 1.0, 0.0; ... % ~SIPI !!! note fake N+R in denom
                
                1.0, 1.0, 1.0, 1.0, 0.0,   1.0, 1.0, 1.0, 1.0, 1.0; ... % "free"
                -1.0, -1.0, -1.0, 1.0, 0.0,   1.0, 1.0, 1.0, 1.0, 1.0; ...  % "free"
                0.0, -1.0, 0.0, 1.0, 0.0,    0.0, 1.0, 0.0, 1.0, 0.0; ...  % ~greenNDVI
                
                0.18, 0.0, 0.0, -1.7, 0.0,   13.7, 0.0, 0.0, 0.45, 0.0;  % dummy
                0.18, 0.0, 0.0, -1.7, 0.0,   13.7, 0.0, 0.0, 0.45, 0.0];  % dummy

num_valid = 10;
alpha_coeffs = alpha_coeffs(1:num_valid,:);
num_ct = size(alpha_coeffs,1);
EPS = 1e-3;  %1e-12;
MAX_VAL = 1e3;

ALL_FILES = 'file_names.mat';
load(ALL_FILES);
num_files = size(FILE_NAMES,1);
BASE_DIR = 'C:\alon\datasets\Agriculture-Vision\';

agct_min_max = cell(num_ct,1);
agct_min_max_all = nan(num_ct,2);
for jj = 1:num_ct
    agct_min_max{jj} = nan(num_files,2);
end
for ii = 1:num_files   % num_files
    disp(ii);
    rgb_file = [];
    nir_file = [];
    RGB = imread([BASE_DIR,FILE_NAMES{ii,1},'\images\rgb\',FILE_NAMES{ii,2},'.jpg']);
    NIR = imread([BASE_DIR,FILE_NAMES{ii,1},'\images\nir\',FILE_NAMES{ii,2},'.jpg']);
    RGB = double(RGB)/255.0;
    NIR = double(NIR)/255.0; 
    for jj = 1:num_ct
        alpha = alpha_coeffs(jj,:);
        [C, minC, maxC] = agct(RGB, NIR, alpha, EPS, MAX_VAL);
        agct_min_max{jj}(ii,:) = [minC,maxC];
    end    
end
for jj = 1:num_ct
    agct_min_max_all(jj,1) = min(agct_min_max{jj}(:,1));
    agct_min_max_all(jj,2) = max(agct_min_max{jj}(:,2));
end
save('agct_min_max','agct_min_max','agct_min_max_all','alpha_coeffs','EPS');
