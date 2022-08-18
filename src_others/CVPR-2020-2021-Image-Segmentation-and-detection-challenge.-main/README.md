                                                            CVPR-2020-2021
# Satellite Earth Image-Segmentation-and-Detection-challenge.

Lets' segment satellite earth images.


![alt text](https://github.com/Deepika-Sharma08/CVPR-2020-2021-Image-Segmentation-and-detection-challenge./blob/main/supporting%20images/1.png?raw=true)


This is an open competition posted by CVPR 2020 & 2021] Agriculture-Vision Dataset, Prize Challenge and Workshop: 

A joint effort with many great collaborators to bring Agriculture and Computer Vision/AI communities together to benefit humanity! 



Resources : 

www.agriculture-vision.com and https://github.com/SHI-Labs/Agriculture-Vision#download.

I have been working on this problem for a few months now and have tried various methods to solve this problem. 
The description of the problem is as follows:
The dataset used in this challenge is a subset of the Agriculture-Vision dataset.
The challenge dataset contains 21,061 aerial farmland images captured throughout 2019 across the US. 
Each image consists of four 512x512 color channels, which are RGB and Near Infra-red (NIR). Each image also has a boundary map and a mask. 
The boundary map indicates the region of the farmland, and the mask indicates valid pixels in the image. 

Regions outside of either the boundary map or the mask are not evaluated.

This dataset contains six types of annotations: Cloud shadow, Double plant, Planter skip, Standing Water, Waterway and Weed cluster. 
These types of field anomalies have great impacts on the potential yield of farmlands, therefore it is extremely important to accurately locate them. 
In the Agriculture-Vision dataset, these six patterns are stored separately as binary masks due to potential overlaps between patterns. 

0 - background,  1 - cloud_shadow,  2 - double_plant,  3 - planter_skip,  4 - standing_water,  5 - waterway,  6 - weed_cluster.

Sample images


![alt text](https://github.com/Deepika-Sharma08/CVPR-2020-2021-Image-Segmentation-and-detection-challenge./blob/main/supporting%20images/2.png?raw=true)



      

      

      

      

      


  

  
No alt text provided for this image
Challenges

   1. Class imbalance.
   2. Poor resolution images.
   3. Class overlap.
   4. Most of the classes as double plant, weed cluster, waterway share similar features.

Resolutions

   1. Removed Planter skip from input data since the images for this class were too less for training.
   2. Filtered images which were poor resolution during pre-processing. Resulted number of images is 12,298 instead 12,901.
   3. For data preparation, I encoded each class with a unique prime number to ensure that each class is uniquely identified, even if classes overlap in image. 
      For  ex: if classes were encoded as 1,2,3,4,5,6 then summing all the images to create one annotated image for training would have entry for overlapped 1 and 5 &     actual 6.
   4. Few more tricks: If classes overlapped, the sum of all the classes would not be one among numbers used for encoding, which is set([1,3,5,11,23,53]).So, I removed overlapping classes since the number of images with overlapping classes turned out to be too less.   
   Re-encoded the classes as 1,2,3,4,5 such that num_of_classes could be input as 6 in the model, else model would pick maximum numeric available in an image as num of classes. 




Training Method

    Trained first model for all 6 classes with following configurations:

a. Model Name : VGG_Unet b. Backbone : ImageNet c. Epochs : 50 d. steps_per_epoch : 5 Reported Loss : 2.5 (which is too high for any classification model)


    #backbone = resnet50

    size = 512

    model = vgg_unet(

           n_classes=6, input_height=size, input_width=size)


    model.train(

       train_images = "input_train/", 

       train_annotations = "train_annotated_final/",n_classes = 6,epochs=50,steps_per_epoch=5,

    )

The accuracy did not look good after trying different backbone nets to train model with 6 classes using various pre processing techniques, since one class is most dominating in the labels and other classes have marginal distribution. So,  I decided to move onto training one model for each class and combine the results to see prediction.


    size = 512

    from keras_segmentation.models.unet import vgg_unet

    model_cloud = vgg_unet(n_classes=2, input_height=size, input_width=size)


    model_cloud.train(

       train_images = "input_train/",

       train_annotations = "train_annotated_cloud_shadow/",n_classes = 2,epochs=10,steps_per_epoch=5,

    )


    Trained model_weed, model_dbl_plant, model_std_water, model_waterway and model_cloud


Finally models result combination

 for idx,img_name in enumerate(os.listdir(input_train)):

out = model_weed/model_dbl_plant/model_cloud/model_std_water/model_waterway. predict_segmentation( inp= dir_+ img_name, out_fname= output_dir + "/" + img_name.split(".")[0] + "_weed/_dbl_plant/_cloud/_std_water/_waterway.png" ) 
Results

To make use of the outcomes of all the trained models (one for each class), I used a histogram based binning to find threshold to identify thresholds for class = 0/1/2/3/4.



 ![alt text](https://github.com/Deepika-Sharma08/CVPR-2020-2021-Image-Segmentation-and-detection-challenge./blob/main/prediction/1.png?raw=true)


![alt text](https://github.com/Deepika-Sharma08/CVPR-2020-2021-Image-Segmentation-and-detection-challenge./blob/main/prediction/2.png?raw=true)


![alt text](https://github.com/Deepika-Sharma08/CVPR-2020-2021-Image-Segmentation-and-detection-challenge./blob/main/prediction/3.png?raw=true)


![alt text](https://github.com/Deepika-Sharma08/CVPR-2020-2021-Image-Segmentation-and-detection-challenge./blob/main/prediction/4.png?raw=true)


![alt text](https://github.com/Deepika-Sharma08/CVPR-2020-2021-Image-Segmentation-and-detection-challenge./blob/main/prediction/5.png?raw=true)


![alt text](https://github.com/Deepika-Sharma08/CVPR-2020-2021-Image-Segmentation-and-detection-challenge./blob/main/prediction/6.png?raw=true)


![alt text](https://github.com/Deepika-Sharma08/CVPR-2020-2021-Image-Segmentation-and-detection-challenge./blob/main/prediction/7.png?raw=true)


![alt text](https://github.com/Deepika-Sharma08/CVPR-2020-2021-Image-Segmentation-and-detection-challenge./blob/main/prediction/8.png?raw=true)

