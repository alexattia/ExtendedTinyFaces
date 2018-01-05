import numpy as np
import dlib
from imgaug import augmenters as iaa
import pandas as pd 
from sklearn.svm import SVC
import random

face_encoder = dlib.face_recognition_model_v1('./model/dlib_face_recognition_resnet_model_v1.dat')
face_pose_predictor = dlib.shape_predictor('./model/shape_predictor_68_face_landmarks.dat')

def encoding_faces(images, label, coord_detect):
    """
    Encoding a list of faces using FaceNet generating a 128D vector
    :param images: list of images of faces to encode
    :param coord_detect: coordinates of the detection
    :return: pandas dataframe of the faces for one label
    """
    l = []
    for img, d in zip(images, coord_detect):
        (x1, y1, x2, y2) = d
        detected_face = dlib.rectangle(left=0, top=0, right=int(x2-x1), bottom=int(y2-y1))
        pose_landmarks = face_pose_predictor(img, detected_face)
        face_encoding = face_encoder.compute_face_descriptor(img, pose_landmarks, 1)
        l.append(np.append(face_encoding, [label]))
    
    return np.array(l)

def create_positive_set(pictures, coords, label=1):
    """
    Create positive train set for one face from a list of three pictures of this face.
    Data Augmentation on these three pictures to generate 10 pictures.
    Encoding of the ten pictures
    :param pictures: list of three full pictures
    :param coords: list of the coordinates of the face in the first frame
    :return: pandas dataframe of the faces for one person
    """
    # original coordinate
    x1, y1, x2, y2 = coords
    # Load the three same faces
    images = [pictures[j][y1:y2,x1:x2,:] for j in range(3)]
    # triple each picture
    images = [item for item in images for i in range(5)]
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    st = lambda aug: iaa.Sometimes(0.5, aug)
    # add a random value from the range (-30, 30) to the first three channels and gaussian noise
    aug = iaa.Sequential([
        iaa.WithChannels( channels=[0, 1, 2], children=iaa.Add((-30, 30))),
        st(iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5))])
    
    images_aug = aug.augment_images(images)
    # add the original frame
    images_aug.append(pictures[0][y1:y2,x1:x2,:])
    
    # encode each of 10 faces (of the same person)
    coords_detect = [coords for k in range(len(images_aug))]
    return pd.DataFrame(encoding_faces(images_aug, label, coords_detect))

def train_binclas(pics, detections, idx_detection):
    """
    Create train set and traing a binary SVM to classify faces for one original
    face (from frame 0) 
    """
    pos = create_positive_set(pics, detections[0][idx_detection])
    
    # Choose 10 other detections from the first frame
    neg_detect = np.array([k for i, k in enumerate(detections[0]) if i != idx_detection])
    idx_neg = random.sample(range(len(neg_detect)), 10)

    # Get face images for the 10 detections
    img_neg = [pics[0][y1_:y2_,x1_:x2_,:] for (x1_, y1_, x2_, y2_) in neg_detect[idx_neg]]
    # Encode each face
    neg = pd.DataFrame(encoding_faces(img_neg, 0, neg_detect[idx_neg]))
    
    # join positive and negative samples
    df = pd.concat([pos, neg])
    df = df.sample(len(df)).reset_index(drop=True)
    y = df[128]
    X = df.drop(128, axis=1)
    
    # training
    clf = SVC(C=1, kernel='linear', probability=True)
    clf.fit(X, y)
    # keeping 4th picture detections in the neighborhoud
    x1, y1, x2, y2 = detections[0][idx_detection] 
    neigh_detect = [k for k in detections[::-1][0] if 
                np.abs(k[0]-x1) < 600 and 
                np.abs(k[1]-y1) < 600 and
                np.abs(k[2]-x2) < 600 and
                np.abs(k[3]-y2) < 600]
    
    # Get face images to classify
    img_neighb = [pics[::-1][0][y1_:y2_,x1_:x2_,:] for (x1_, y1_, x2_, y2_) in neigh_detect]
    # Encode each face
    neigh_detect_encodings = encoding_faces(img_neighb, -1, neigh_detect)[:,:128]
    # compute distances
    distances = clf.predict_proba(neigh_detect_encodings)
   
    return neigh_detect, distances
