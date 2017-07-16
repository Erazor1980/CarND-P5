import HOG_features as hogF
import color_features as colF

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split


def extract_HOG_andColor_features(imgs, cspace='RGB',
                                  # HOG parameters:
                                  orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                                  # Color parameters:
                                  use_spatial=True, use_hist=True,
                                  spatial_size=(32, 32), hist_bins=32, hist_range=(0, 256) ):

    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        ###################################
        #### EXTRACT HOG FEATURES
        ###################################
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(hogF.get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = hogF.get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        curr_features = hog_features
        ###################################
        #### EXTRACT COLOR FEATURES
        ###################################
        if use_spatial == True:
            spatial_features = colF.bin_spatial(feature_image, size=spatial_size)
            curr_features = np.concatenate(( curr_features, spatial_features ))
        if use_hist == True:
            hist_features = colF.color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
            curr_features = np.concatenate((curr_features, hist_features))

        features.append(curr_features)

    # Return list of feature vectors
    return features


#########################################
#### DIVIDE UP INTO CARS AND NOTCARS ####
#########################################
images = glob.glob('./vehicles_smallset/*/*.jpeg') + glob.glob('./non-vehicles_smallset/*/*.jpeg')
cars = []
notcars = []
for image in images:
    if 'image' in image or 'extra' in image:
        notcars.append(image)
    else:
        cars.append(image)

############################
#### FEATURE EXTRACTION ####
############################
# HOG params
colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 0 # Can be 0, 1, 2, or "ALL"
# Color params
spatial_size=(32, 32)
hist_bins=32
hist_range=(0, 256)

t=time.time()
print("Extracting features...")
car_features = extract_HOG_andColor_features(cars, cspace=colorspace, orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block, hog_channel=hog_channel,
                            spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range)
notcar_features = extract_HOG_andColor_features(notcars, cspace=colorspace, orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block, hog_channel=hog_channel,
                            spatial_size=spatial_size, hist_bins=hist_bins, hist_range=hist_range)
t2 = time.time()
print("...done in ", round(t2-t, 2), 'Seconds.')

############################################
#### NORMALIZING DATA AND ADDING LABELS ####
############################################
# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

#############################################
#### SPLITTING DATA IN TRAINING AND TEST ####
#############################################
# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
print('Using:',orient,'orientations',pix_per_cell, 'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))

##########################################
#### CREATING AND TRAINING CLASSIFIER ####
##########################################
#TODO maybe add parameter tuning (if results are not good enough!)
#parameters = {'kernel':('linear', 'poly', 'rbf'), 'C':[1, 10]}
#svr = svm.SVC()
#clf = grid_search.GridSearchCV(svr, parameters)
#clf.fit(iris.data, iris.target)

# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

#### small test on 20 images
n_predict = 20
print('My SVC predicts: \n', svc.predict(X_test[0:n_predict]))
print('For these',n_predict, 'labels: \n', y_test[0:n_predict])
t2 = time.time()
print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')