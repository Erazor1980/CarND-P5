from hog_subsample import *
from helper import *
from scipy.ndimage.measurements import label
import pickle

test_on_single_image = False

filename = 'svc_pickle.p'
data = pickle.load(open(filename, 'rb'))

X_scaler        = data['X_scaler']
svc             = data['svc']
orient          = data['orient']            # HOG orientations
pix_per_cell    = data['pix_per_cell']      # HOG pixels per cell
cell_per_block  = data['cell_per_block']    # HOG cells per block
spatial_size    = data['spatial_size']      # Spatial binning dimensions
hist_bins       = data['hist_bins']         # Number of histogram bins
ystart = 400
ystop = 656
scale = 1.5

#image = mpimg.imread('./test_images/test6.jpg')
#box_list, out_img = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
 #                   hist_bins)
#plt.imshow(out_img)
#plt.show()

def process_frame(image, video=True):
    box_list, _ = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,
                         cell_per_block, spatial_size, hist_bins)

    box_list2, _ = find_cars(image, ystart, ystop, 1, svc, X_scaler, orient, pix_per_cell,
                         cell_per_block, spatial_size, hist_bins)
    box_list += box_list2


    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,box_list)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,1)
    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    if video == True:
        return draw_img
    else:
        return draw_img, heatmap


if test_on_single_image == True:
    image = mpimg.imread('./test_images/test6.jpg')
    draw_img, heatmap = process_frame(image, False)
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(draw_img)
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()
    plt.show()
else:
    #######################################
    #### LOADING AND PROCESSING VIDEO  ####
    #######################################
    import imageio
    imageio.plugins.ffmpeg.download()
    from moviepy.editor import VideoFileClip

    result_path = 'output/result_video_test.mp4'
    video = VideoFileClip("test_video.mp4")
    result_video = video.fl_image(process_frame) #NOTE: this function expects color images!!
    result_video.write_videofile(result_path, audio=False)

