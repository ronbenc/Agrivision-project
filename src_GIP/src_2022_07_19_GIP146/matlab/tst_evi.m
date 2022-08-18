clear all; close all;

BASE_DIR = 'C:\alon\datasets\Agriculture-Vision\';
IMG_SIZE = [512,512];
CASE_DIR = 'val\';  %%% 'val\'  'train\'    'test\'
RGB_DIR = [BASE_DIR,CASE_DIR,'images\rgb\'];
NIR_DIR = [BASE_DIR,CASE_DIR,'images\nir\'];


f_name = '88RUIM9H6_1488-5949-2000-6461'; 
RGB = imread([RGB_DIR,f_name,'.jpg']); 
NIR = imread([NIR_DIR,f_name,'.jpg']); 
R = double(RGB(:,:,1)) / 255.0;
NIR = double(NIR) / 255.0;   
nomin = (NIR - R);
denom = (NIR + R);
NDVI = nomin ./ denom;
NDVI(denom==0) = 0;
NDVI = uint8(127.5 * (NDVI + 1)); 

B = double(RGB(:,:,3)) / 255.0;
G = 2.5;
C1 = 6.0;
C2 = 7.5;
L = 1.0;
min_evi = -10.0;
max_evi = 10.0;
denom = NIR + C1*R - C2*B + L; 
EVI = G * (nomin ./ denom);
EVI(denom==0) = 0;
% EVI(EVI < min_evi) = min_evi;
% EVI(EVI > max_evi) = max_evi;
figure;
subplot(2,2,1); imshow(RGB); title('RGB');
subplot(2,2,2); imshow(NIR,[]); title('NIR');
subplot(2,2,3); imshow(NDVI,[]); title('NDVI');
subplot(2,2,4); imshow(EVI,[-5,5]); title('EVI');



