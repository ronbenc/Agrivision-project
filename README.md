# Agrivision-project

This repo in cloned from https://github.com/samleoqh/MSCG-Net.
All right reserved to the authors of the original article.
https://openaccess.thecvf.com/content_CVPRW_2020/papers/w5/Liu_Multi-View_Self-Constructing_Graph_Convolutional_Networks_With_Adaptive_Class_Weighting_Loss_CVPRW_2020_paper.pdf


# Agriculture-Vision dataset and contest
https://www.agriculture-vision.com/agriculture-vision-2020/dataset

https://github.com/SHI-Labs/Agriculture-Vision


## Contribution (only in src_mor_ron/MSCG_Net):
- added support for weights and biases (wandb)
- Added agricultural indices augmentation module (learned and static)

* before onboarding checkout the MSCG-Net readme file and follow its configuration instructions.

## Onboarding:
1. create a conda invironment and download all the needed packages (in requirements.txt)
2. download the dataset and set the path to it in the needed files (see MSCG-Net readme file and follow the instructions there)
3. for running with wandb to log your runs, first login to wandb , in train_R50.py and train_R101.py in wandb.init change the entity to yours.  

## How to run:
- in this example we train the model with a backbone of a ResNet-101 augment with a NDVI channel.
- python tools/train_R101.py NDVI_r101_run  --NDVI

- in order to log into wandb we would add the --wandb flag
python tools/train_R101.py --wandb  --NDVI

- in order to log into wandb under a specific run name we would add the --run_name <name> flag
python tools/train_R101.py --wandb --run_name <name>  --NDVI

* for running with wandb you must:
1. log in to your wandb account
2. change the entity name in wandb.init to your username

- in this example we train the model with a backbone of a ResNet-50 augmented with learnable indices (initialized with EVI, GCC)+ static indices (SAVI gNDVI) 
python tools/train_R50.py --learn EVI --learn GCC --SAVI --gNDVI

## How to generate a submission:
1. set the paths to the saved models snapshots you selected in ckpt.py
2. run: python test_submission.py 
