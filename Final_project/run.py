colab = False
if colab:
    from google.colab import drive
    drive.mount('/content/drive')
    data_dir = '/content/drive/My Drive/Final_project/data'
    model_dir = '/content/drive/My Drive/Final_project/model'
    output_dir = '/content/drive/My Drive/Final_project/output'
else:
    data_dir = './data'
    model_dir = './model'
    output_dir = './output'




from torchvision import datasets, models, transforms
import torch.nn as nn
import numpy as np
import torchvision
import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from scipy.io import loadmat
from torch.autograd import Variable
import cv2
from sklearn.svm import LinearSVC
# from matplotlib import pyplot as plt
import torch.nn.functional as F
    
use_GPU = torch.cuda.is_available()
if use_GPU:
    print("Using CUDA")
    device = torch.device("cuda") 
else:
    print ("Using CPU")


    
transform = transforms.Compose([
                transforms.Resize([48, 48]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])]) # orignally for ImageNet



import torch.nn.functional as F

drop_rate = 0.2

class my_model(nn.Module):
    
    def __init__(self):
        super(my_model, self).__init__()

        self._hidden1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(drop_rate)
        )
        self._hidden2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
            nn.Dropout(drop_rate)
        )
        self._hidden3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(drop_rate)
        )
        self._hidden4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=160),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
            nn.Dropout(drop_rate)
        )
        self._hidden5 = nn.Sequential(
            nn.Conv2d(in_channels=160, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(drop_rate)
        )
        self._hidden6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
            nn.Dropout(0.2)
        )
        self._hidden7 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(drop_rate)
        )
        self._hidden8 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=192),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
            nn.Dropout(drop_rate)
        )
        self._hidden9 = nn.Sequential(
            nn.Linear(192 * 3 * 3, 3072),
            nn.ReLU()
        )
        self._hidden10 = nn.Sequential(
            nn.Linear(3072, 3072),
            nn.ReLU()
        )

        self._output = nn.Sequential(nn.Linear(3072, 11))

    def forward(self, x):
        x = self._hidden1(x)
        x = self._hidden2(x)
        x = self._hidden3(x)
        x = self._hidden4(x)
        x = self._hidden5(x)
        x = self._hidden6(x)
        x = self._hidden7(x)
        x = self._hidden8(x)
        x = x.view(x.size(0), 192 * 3 * 3)
        x = self._hidden9(x)
        x = self._hidden10(x)

        output = self._output(x)  
        
        return output

# test = my_model()
# print(test)



# VGG16

pretrained = False
if pretrained:
    # load pretrained model
    vgg16 = models.vgg16(pretrained=True) # pretrained
    
    for param in vgg16.features.parameters():
        param.require_grad = False # Freeze training for all layers
else:
    vgg16 = models.vgg16() # random weights

# modify the output of the last layer from 1000 to 11
num_classes = 11 # 10 digits and null
num_in_features = vgg16.classifier[6].in_features # totally 7 modules inside
features = list(vgg16.classifier.children())[:-1] # Remove last layer
# Newly created modules have require_grad=True by default
features.extend([nn.Linear(num_in_features, num_classes)]) # Add last layer with 11 outputs
vgg16.classifier = nn.Sequential(*features) # Replace the last layer of classifier

# print(vgg16)





# produce 5 images

# using MSER

def MSER_detect(img, delta, min_area, max_area):
    
    image = np.copy(img)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5,5),0) # blur

    mser = cv2.MSER_create(_delta=delta, _min_area = min_area, _max_area = max_area)
    regions = mser.detectRegions(img_blur)
#     regions = mser.detectRegions(img_gray)
    
    color = (255, 0, 0)
    thickness = 20
    for region in regions[1]:
        start_point = (region[0], region[1])
        end_point = (region[0] + region[2], region[1] + region[3])
        cv2.rectangle(image, start_point, end_point, color, thickness)

#     plt.imshow(image)
#     plt.show()

    return regions[1]

    
# I would recommend to use the raw logits + nn.CrossEntropyLoss for training 
# and if you really need to see the probabilities, 
# just call F.softmax on the output as described in the other post.

def predict(model, images):
    model.train(False)
    model.eval()
    
    print ()
    print ('start prediction')

    if use_GPU:
        images = images.to(device)
    
    model.zero_grad()
    outputs = model(images)

    _, preds = torch.max(outputs.data, 1)
    probabilities = F.softmax(outputs,dim=1)
    scores = torch.max(probabilities, dim=1)[0]

#     del images, outputs, preds, probability
#     torch.cuda.empty_cache()
    
    return preds, scores


# enlarger ratio is how much to enlarge the box.
# wh ratio is width/height ratio range, 0.5-2.0

def create_subimages_with_scores(image, regions,
                                 enlarger_ratio = 1.0, wh_ratio = 2.0,
                                 min_w = 30, min_h = 30, 
                                 max_w = 1000, max_h = 1000) :
    boxes = []
    subimages = []

#     image = cv2.imread(os.path.join(data_dir, image_file))
    img = np.copy(image)
    print (img.shape)

    for box in regions:
        x1 = int(box[0]-box[2]*1.0)
        y1 = int(box[1]-box[3]*1.0)
        x2 = int(box[0] + box[2] + box[2]*1.0)
        y2 = int(box[1] + box[3] + box[3]*1.0)
        
        # if the point location is outside of the image, keep it on the edge.
        if x1 < 0: 
            x1 = box[0]
        if y1 < 0:
            y1 = box[1]
        if x2 > img.shape[1]:
            x2 = box[0] + box[2]
        if y2 > img.shape[0]:
            y2 = box[1] + box[3]

        if box[2] >= min_w and box[3] >= min_h and box[2] <= max_w and box[3] <= max_h and box[3]/box[2]<=wh_ratio and box[2]/box[3]<=wh_ratio:
            boxes.append((x1,y1,x2,y2))
            image_cut = img[y1:y2,x1:x2,:] 
#             print ((x1,y1,x2,y2))
            print ()
            image_cut = torchvision.transforms.functional.to_pil_image(image_cut)      
            image_cut = transform(image_cut)        
            subimages.append(image_cut)
    
    

    subimages = torch.stack(subimages)
    labels, scores = predict(detect_model, subimages)

    print (labels)
    print (scores)

    # remove boxes with label 10
    new_boxes = []
    new_scores = []
    new_labels = []
    for i, label in enumerate(labels):
        if label != 10:
            new_boxes.append(boxes[i])
            new_scores.append(scores[i].item())
            new_labels.append(labels[i].item())

    print (new_boxes)
    print (new_scores)
    print (new_labels)

    color = (255, 0, 0)
    thickness = 20
    for box in new_boxes:
        start_point = (box[0], box[1])
        end_point = (box[2], box[3])
        cv2.rectangle(img, start_point, end_point, color, thickness)

#     plt.imshow(img)
#     plt.show()
    
    return new_boxes, new_labels, new_scores



def merge_boxes_and_digits(image, boxes, labels, scores, keep_idx):
    
    
    
    remove_idx = []
    for i in keep_idx:
        for j in keep_idx:
            if boxes[i][0]<boxes[j][0] and boxes[i][1]<boxes[j][1] and boxes[i][2]>boxes[j][2] and boxes[i][3]>boxes[j][3]:
                remove_idx.append(j)
    
    keep_idx = keep_idx.numpy().tolist()       
    for idx in remove_idx:
        keep_idx.remove(idx)
    
    # if the centers of two clusters are far than some distance, then 
    
    keep_box = np.array([boxes[idx] for idx in keep_idx])
    keep_numbers = np.array([labels[idx] for idx in keep_idx])
    keep_scores = np.array([scores[idx] for idx in keep_idx])
    
    
    Z = np.array(keep_box)
    print (Z)
    
    if len(Z)>1:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
        # (type,max_iter,epsilon)
        flags=cv2.KMEANS_RANDOM_CENTERS
        compactness,label,center = cv2.kmeans(np.float32(Z), 2, None, criteria, 10, flags)

        print (label)
        print ("label = ", label.ravel)
        C = Z[label.ravel()==0]
        D = Z[label.ravel()==1]
        print (C)
        print (D)

        print ("center = ", center)

        # cluster_a_center = 

        print ("keep_scores")
        print (keep_scores)

        if abs(center[0][1]+center[0][3]-center[1][1]-center[1][3])/2.0>=0.1*image.shape[0]:
            # two clusters
            A = np.mean(keep_scores[label.ravel()==0])
            B = np.mean(keep_scores[label.ravel()==1])
            if A > B:
                keep_box = keep_box[label.ravel()==0]
                keep_numbers = keep_numbers[label.ravel()==0]
                keep_scores = keep_scores[label.ravel()==0]
            else:
                keep_box = keep_box[label.ravel()==1]
                keep_numbers = keep_numbers[label.ravel()==1]
                keep_scores = keep_scores[label.ravel()==1]
    
    
    x_centers = [(keep_box[idx][0]+keep_box[idx][2])/2 for idx in range(len(keep_box))]    
    digit_order = np.argsort(x_centers)
    digits = [str(keep_numbers[i]) for i in digit_order]
    final_number = int("".join(digits))
    
    final_x1 = np.min([keep_box[idx][0] for idx in range(len(keep_box))])
    final_y1 = np.min([keep_box[idx][1] for idx in range(len(keep_box))])
    final_x2 = np.max([keep_box[idx][2] for idx in range(len(keep_box))])
    final_y2 = np.max([keep_box[idx][3] for idx in range(len(keep_box))])

    print (str(final_number))
    
#     final_img = np.copy(image)
#     color = (255, 0, 0)
#     thickness = 20
#     cv2.rectangle(final_img, (final_x1, final_y1), (final_x2, final_y2), color, thickness)
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.putText(final_img, str(final_number), (int((final_x1+final_x2)/2), final_y1-50), font, 8, color, 30)
#     plt.imshow(final_img)
#     plt.show()
    
    return final_number, (final_x1, final_y1, final_x2, final_y2)


    
def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Open file with VideoCapture and set result to 'video'.
    video = cv2.VideoCapture(filename)

    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Close video (release) and yield a 'None' value.
    video.release()
    yield None

    
def mp4_video_writer(filename, frame_size, fps=20):
    """Opens and returns a video for writing.

    Use the VideoWriter's `write` method to save images.
    Remember to 'release' when finished.

    Args:
        filename (string): Filename for saved video
        frame_size (tuple): Width, height tuple of output video
        fps (int): Frames per second
    Returns:
        VideoWriter: Instance of VideoWriter ready for writing
    """
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    return cv2.VideoWriter(filename, fourcc, fps, frame_size)    

    

def detect_and_plot_numbers(image, font_size=6, thickness=20):
    
#     regions = MSER_detect(image, 4, 1000, 5000)
    
    regions = MSER_detect(image, 4, 1000, 50000)

    print (len(regions))
    
    if len(regions) > 0: 

        boxes, labels, scores = create_subimages_with_scores(image,regions, 
                                         enlarger_ratio = 0.5, wh_ratio = 2.0,
                                         min_w = 30, min_h = 30, 
                                         max_w = 1000, max_h = 1000)
        
        if len(boxes) > 0:
        
            # nms
            iou_threshold = 0.5
            keep_idx = torchvision.ops.nms(torch.FloatTensor(boxes), torch.FloatTensor(scores), iou_threshold)
            print ("keep_idx = ", keep_idx)

            if len(keep_idx.numpy()) > 0:

                # merge boxes and generate digits  
                number, (x1, y1, x2, y2) = merge_boxes_and_digits(image, boxes, labels, scores, keep_idx)

                # generate image
                color = (255, 0, 0)
                thickness = 20
                cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, str(number), (int((x1+x2)/2), y1-50), font, font_size, color, thickness)
#                 plt.imshow(image)
#                 plt.show()




def generate_video(VID_DIR, video_name, OUTPUT_DIR, filename, fps, frame_ids):
    
    video = os.path.join(VID_DIR, video_name+".mp4")
    print (video)
    image_gen = video_frame_generator(video)

    image = image_gen.__next__()
    h, w, d = image.shape

    out_path = os.path.join(OUTPUT_DIR, video_name+"_with_numbers.mp4")
    
    video_out = mp4_video_writer(out_path, (w, h), fps)
    
    frame_num = 1
    

    while image is not None:

        print("Processing fame {}".format(frame_num))
        
        detect_and_plot_numbers(image, font_size=6, thickness=20)
        
#         if frame_num in frame_ids:      
#             cv2.imwrite(os.path.join(OUTPUT_DIR, str(filename) + '_'+ str(frame_num) + ".png"), image)
        
        video_out.write(image)
        
#         cv2.imshow('image', image)
        
#         if cv2.waitKey(1) == ord('q'):
#             break

        image = image_gen.__next__()

        frame_num += 1
        
#         if frame_num >= 280:
        
#             image = image[140:, 250:, :]
#             image = cv2.resize(image, (w, h))

    video_out.release()
    cv2.destroyAllWindows()
    

model_save_name = 'MyModel_SVHN_Dec_03.pt'

if use_GPU:
    model_path = F'/content/drive/My Drive/Final_project/model/{model_save_name}' 
    detect_model = my_model()
    detect_model.load_state_dict(state_dict=torch.load(model_path))
    my_model.cuda()
    print('My model loaded!') 

else:
    model_path = os.path.join(model_dir, model_save_name)
    detect_model = my_model()
    detect_model.load_state_dict(state_dict=torch.load(model_path, map_location=torch.device('cpu')))
    print('My model loaded!') 







# # extract frames
# def extract_frames():
#     video_name = "IMG_2890"
#     filename = "IMG_2890"

#     frame_ids = [15, 21, 176, 177]

#     video = os.path.join(data_dir, video_name+".mp4")
#     image_gen = video_frame_generator(video)

#     image = image_gen.__next__()
#     h, w, d = image.shape

#     frame_num = 1

#     while image is not None:

#         if frame_num in frame_ids:  
#             print("Processing fame {}".format(frame_num))
#             cv2.imwrite(os.path.join(output_dir, str(filename) + '_'+ str(frame_num) + ".png"), image)

#         image = image_gen.__next__()

#         frame_num += 1

        
        
        
        
# produce 5 images

# original

frame_name = "IMG_2890_15.png"
save_name = "0.png"
image = cv2.imread(os.path.join(data_dir, frame_name))
detect_and_plot_numbers(image, font_size=6, thickness=20)
cv2.imwrite(os.path.join(output_dir, save_name), image)


# brightness

frame_name = "IMG_2890_21.png"
save_name = "1.png"
image = cv2.imread(os.path.join(data_dir, frame_name))
image = (image*0.75).astype(np.uint8)
detect_and_plot_numbers(image, font_size=6, thickness=20)
cv2.imwrite(os.path.join(output_dir, save_name), image)


# add noise

frame_name = "IMG_2890_21.png"
save_name = "2.png"
image = cv2.imread(os.path.join(data_dir, frame_name))

m = (0,0,0) 
s = (20,20,20)
temp = np.zeros_like(image)
noise = cv2.randn(temp,m,s);

image = image + noise

detect_and_plot_numbers(image, font_size=6, thickness=20)
cv2.imwrite(os.path.join(output_dir, save_name), image)


# amplification

frame_name = "IMG_2890_21.png"
save_name = "3.png"
image = cv2.imread(os.path.join(data_dir, frame_name))
h, w, d = image.shape
image = image[125:875, 293:1627, :]
image = cv2.resize(image, (w, h))
print (image.shape)
detect_and_plot_numbers(image, font_size=6, thickness=20)
cv2.imwrite(os.path.join(output_dir, save_name), image)

# location

frame_name = "IMG_2890_21.png"
save_name = "4.png"
image = cv2.imread(os.path.join(data_dir, frame_name))
h, w, d = image.shape
image = image[:800, 350:, :]
# image = cv2.resize(image, (w, h))
print (image.shape)
detect_and_plot_numbers(image, font_size=6, thickness=20)
cv2.imwrite(os.path.join(output_dir, save_name), image)



# rotate

frame_name = "IMG_2890_176.png"
save_name = "5.png"
image = cv2.imread(os.path.join(data_dir, frame_name))
detect_and_plot_numbers(image, font_size=6, thickness=20)
cv2.imwrite(os.path.join(output_dir, save_name), image)









