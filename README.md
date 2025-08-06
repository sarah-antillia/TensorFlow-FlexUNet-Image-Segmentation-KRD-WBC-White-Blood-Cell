<h2>TensorFlow-FlexUNet-Image-Segmentation-KRD-WBC-White-Blood-Cell (2025/08/07)</h2>

This is the first experiment of Image Segmentation for KRD-WBC White Blood Cell Multiclass,
 based on our 
TensorFlowFlexUNet (TensorFlow Flexible UNet Image Segmentation Model for Multiclass) 
and a 512x512 pixels 
<a href="https://drive.google.com/file/d/1sriI2gBrpCYVOTLpndlXPK3VQnCgthWL/view?usp=sharing">
<b>Augmented-KRD-WBC-ImageMask-Dataset.zip</b></a>
with colorized masks (Neutrophil:green, Lymphocyte:blue, Monocyte:red, Eosinophil:cyan, Basophil:yellow).
which was derived by us from 
<a href="https://data.mendeley.com/datasets/jzdj6h7gms/2">
<b>
Creating a white blood cell dataset for segmentation
</b>
</a>
<br>
<br>
<b>Acutual Image Segmentation for 512x512 KRD-WBC images</b><br>

As shown below, the inferred masks predicted by our segmentation model, which was trained on 
the augmented dataset, 
appear similar to the ground truth masks in shape, but differ in color.
<br>



<br>

<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test/images/1017.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test/masks/1017.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test_output/1017.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test/images/1025.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test/masks/1025.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test_output/1025.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test/images/1083.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test/masks/1083.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test_output/1083.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1. Dataset Citation</h3>
The dataset used here has been taken from the web-site:<br>
<a href="https://data.mendeley.com/datasets/jzdj6h7gms/2">
<b>
Creating a white blood cell dataset for segmentation
</b>
</a>
<br>
<br>
<b>Published:</b>  9 August 2023 | Version 2 | DOI: 10.17632/jzdj6h7gms.2<br>
<b>Contributors:</b> Haval Taha, Fattah Alizadeh, Nawsherwan Mohammad
<br>
<br>
<b>Description</b><br>
The KRD-WBC dataset consists of 600 images, each sized at 512x512 pixels, accompanied by corresponding ground truth 
images of the same dimensions. The dataset was gathered from Nanakali and Bio lab in Erbil city, 
located in the Kurdistan region of Iraq. It includes images of White Blood Cells (WBCs) 
categorized into five distinct subtypes: Neutrophils, Lymphocytes, Monocytes, Eosinophils, and Basophils. 
This dataset is valuable for medical and biological research, enabling the study and analysis of different WBC subtypes.
<br>
<br>
<b>Licence</b><br>
CC BY 4.0 
<br>
<br>

<h3>
<a id="2">
2 KRD-WBC ImageMask Dataset
</a>
</h3>
 If you would like to train this KRD-WBC Segmentation model by yourself,
 please download the dataset from the google drive  
<a href="https://drive.google.com/file/d/1sriI2gBrpCYVOTLpndlXPK3VQnCgthWL/view?usp=sharing">
<b>Augmented-KRD-WBC-ImageMask-Dataset.zip</b></a>.
<br>
, expand the downloaded ImageMaskDataset and put it under <b>./dataset</b> folder to be
<pre>
./dataset
└─KRD-WBC
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>KRD-WBC Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/KRD-WBC/KRD-WBC_Statistics.png" width="512" height="auto"><br>
<br>
<!--
On the derivation of the dataset, please refer to the following Python scripts:<br>
<li><a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a></li>
<li><a href="./generator/split_master.py">split_master.py</a></li>
<br>
-->
As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br>
<br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/KRD-WBC/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/KRD-WBC/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<h3>
3 Train TensorFlowFlexUNet Model
</h3>
 We trained KRD-WBC TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/KRD-WBC/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/KRD-WBC and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16 </b> and large <b>base_kernels = (9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
;You may specify your own UNet class derived from our TensorFlowFlexModel
model         = "TensorFlowFlexUNet"
generator     =  False
image_width    = 512
image_height   = 512
image_channels = 3
num_classes    = 6

base_filters   = 16
base_kernels   = (9,9)
num_layers     = 8
dropout_rate   = 0.04
dilation       = (1,1)
</pre>
<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>
<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Dataset class</b><br>
Specifed <a href="./src/ImageCategorizedMaskDataset.py">ImageCategorizedMaskDataset</a> class.<br>
<pre>
[dataset]
class_name    = "ImageCategorizedMaskDataset"
</pre>
<br>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for KRD-WBC 1+3 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
;
; RGB colors  (Neutrophil:green, Lymphocyte:blue, Monocyte:red, Eosinophil:cyan, Basophil:yellow)
rgb_map = {(0,0,0):0,(0,255,0):1,   (0,0, 255):2,    (255, 0, 0):3,  (0,255, 255);4,  (255,255,0):5 }



</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInfereuncer.py">epoch_change_infer callback</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/KRD-WBC/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 14,15,16)</b><br>
<img src="./projects/TensorFlowFlexUNet/KRD-WBC/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (epoch 29,30,31)</b><br>
<img src="./projects/TensorFlowFlexUNet/KRD-WBC/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>
In this experiment, the training process was terminated at epoch 31.<br><br>
<img src="./projects/TensorFlowFlexUNet/KRD-WBC/asset/train_console_output_at_epoch31.png" width="920" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/KRD-WBC/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/KRD-WBC/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/KRD-WBC/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/KRD-WBC/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to <b>./projects/TensorFlowFlexUNet/KRD-WBC</b> folder,<br>
and run the following bat file to evaluate TensorFlowFlexUNet model for KRD-WBC.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer_aug.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/KRD-WBC/asset/evaluate_console_output_at_epoch31.png" width="920" height="auto">
<br><br>

<a href="./projects/TensorFlowFlexUNet/KRD-WBC/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this KRD-WBC/test was low and dice_coef_multiclass 
high as shown below.
<br>
<pre>
categorical_crossentropy,0.0188
dice_coef_multiclass,0.9917
</pre>
<br>

<h3>
5 Inference
</h3>
Please move <b>./projects/TensorFlowFlexUNet/KRD-WBC</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for KRD-WBC.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer_aug.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/KRD-WBC/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/KRD-WBC/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/KRD-WBC/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks of 512x512 pixels</b><br>
RGB-map: (Neutrophil:green, Lymphocyte:blue, Monocyte:red, Eosinophil:cyan, Basophil:yellow).<br>
As shown below, the inferred masks predicted by our segmentation model, which was trained on the 
augmented dataset, 
appear similar to the ground truth masks in shape, but differ in color.<br>
<table>
<tr>
<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test/images/1017.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test/masks/1017.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test_output/1017.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test/images/1024.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test/masks/1024.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test_output/1024.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test/images/1086.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test/masks/1086.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test_output/1086.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test/images/1114.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test/masks/1114.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test_output/1114.png" width="320" height="auto"></td>
</tr>



<tr>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test/images/barrdistorted_1001_0.3_0.3_1198.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test/masks/barrdistorted_1001_0.3_0.3_1198.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test_output/barrdistorted_1001_0.3_0.3_1198.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test/images/1355.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test/masks/1355.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/KRD-WBC/mini_test_output/1355.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>

<b>1.Creating a white blood cell dataset for segmentation</b><br>

<a href="https://data.mendeley.com/datasets/jzdj6h7gms/2">
https://data.mendeley.com/datasets/jzdj6h7gms/2
</a>
<br>
