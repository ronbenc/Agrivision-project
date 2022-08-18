%clear all; %close all;

BASE_DIR = 'C:\alon\datasets\Agriculture-Vision\';
CASE_DIR = 'val\';  %%% 'val\'  'train\'    'test\'
prefix = 'B87ABMJUV_1781-2479-2293-2991';
IMG_SIZE = [512,512];


RGB_DIR = 'images\rgb\';
NIR_DIR = 'images\nir\';
LABEL_NAMES = { 'boundaries\' ...
                'masks\', ...
                'labels\cloud_shadow\', ...
                'labels\double_plant\', ...
                'labels\planter_skip\', ...
                'labels\standing_water\', ...
                'labels\waterway\', ...
                'labels\weed_cluster\'};
            


rgb = imread([BASE_DIR,CASE_DIR,RGB_DIR,prefix,'.jpg']);
nir = imread([BASE_DIR,CASE_DIR,NIR_DIR,prefix,'.jpg']);
I = uint8(zeros(IMG_SIZE(1),IMG_SIZE(2),8));
for ii = 1:8
    I(:,:,ii) = imread([BASE_DIR,CASE_DIR,LABEL_NAMES{ii},prefix,'.png']);
end

figure;
subplot(2,5,1);
imshow(rgb);impixelinfo;
title('RGB');
subplot(2,5,2);
imshow(nir);impixelinfo;
title('NIR');
sgtitle(prefix,'Interpreter','None');
for ii = 1:8
    subplot(2,5,ii+2);
    imshow(I(:,:,ii));impixelinfo;
    title(LABEL_NAMES{ii},'Interpreter','None');
end



