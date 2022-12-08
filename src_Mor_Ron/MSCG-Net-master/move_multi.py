import os
import shutil

# this scripts purpose is to remove duplicates of images in the dataset
DIR = '/home/ronbenc/Agrivision-project/dataset/train/images/nir'
multi_dir = '/home/ronbenc/Agrivision-project/dataset/train/images/train_multi_nir'
os.mkdir(multi_dir)
count=0
total_count=0
for parent, dirnames, filenames in os.walk(DIR):
    for fn in filenames:
        total_count +=1 
        if "multi" in fn:
            print(fn)
            print(os.path.join(parent, fn))
            shutil.move(os.path.join(parent, fn), os.path.join(multi_dir, fn))
            count+= 1
print("count: " , count)
print("total_count: " , total_count)

DIR = '/home/ronbenc/Agrivision-project/dataset/val/images/nir'
multi_dir = '/home/ronbenc/Agrivision-project/dataset/val/images/train_multi_nir'
os.mkdir(multi_dir)
count=0
total_count=0
for parent, dirnames, filenames in os.walk(DIR):
    for fn in filenames:
        total_count +=1 
        if "multi" in fn:
            print(fn)
            print(os.path.join(parent, fn))
            shutil.move(os.path.join(parent, fn), os.path.join(multi_dir, fn))
            count+= 1
print("count: " , count)
print("total_count: " , total_count)

