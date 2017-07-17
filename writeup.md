## Writeup
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output/car_not_car.png
[image2]: ./output/sliding_window.png
[image3]: ./output/sliding_window2.png
[image4]: ./output/heat_map.png
[image5]: ./output/heat_map_result_bb.png

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it!

### Histogram of Oriented Gradients (HOG)

The code for this step is contained in lines 108 through 140 of the file called `hog_subsample.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Next I explored different color spaces and parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I used the `extract_features` function located in `helper.py` to try out several parameter combinations.
I tried using only HOG features, then added color features (see `bin_spatial` and `color_hist` in `helper.py`).
I reached best results with following combination of parameters:

| Parameter     | Value| 
|:-------------:|:-------------:| 
| orient        | 9        | 
| pix_per_cell  | 8      |
| cell_per_block| 2      |
| spatial_size  | (32, 32)        |
| color space | LUV      |

Feature extraction is located in lines 142 through 146 of `hog_subsample.py`. Features were scaled using StandardScalar (see lines 150-154).
I trained a linear SVM using parameters shown in the table above (lines 168-171). Feature vector length was 8460, consisting of spatial features, histogram features and HOG features. The data was divided into training and test sets using train_test_split from sklearn.model_selection (see lines 160-162, 20% test, 80% training).


### Sliding Window Search

I decided to search only in relevant area of the image (ystart = 400, ystop 656) with 2 different scale sizes (1.5 and 1). For the steps in x and y direction I chose 2 cells (see line 40 in `hog_subsample.py`), so there is a huge overlap. An example image is presented below:

![alt text][image2]

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here another example image:

![alt text][image3]
---

### Video Implementation

Here's a [link to my video result](./output/result_video.mp4)

I recorded the positions of positive detections in each frame of the video (the function find_cars returns bounding boxes of detected cars, see line 93 in `hog_subsample.py`). From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle. I constructed bounding boxes to cover the area of each blob detected. All appropriate functions can be found in `helper.py`.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is a frames and its corresponding heatmap:

![alt text][image4]

### Here the resulting bounding boxes:
![alt text][image5]

---

###Discussion

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

To improve the algorithm I would perform following steps:

* Parameter tuning using Cross-validation with GridSearchCV, to find the best parameter combination for the classifier automatically
* Adding more scales to find bigger/smaller cars (nearer and more far away)
* Cumulative heatmap over n last frames to make the algo more robust (and remove even more outliers)

