import numpy as np
import pandas as pd
import glob
data_folder = './tiny/data/widerface/WIDER_val/images/'

def get_folder_name(pic):
    """
    Get folder name from the picture name
    1_Handshaking_Handshaking_1_411.jpg -->1--Handshaking/
    :param pic: picture name
    :return: folder name
    """
    x = pic.split('_')[1:3]
    s = pic.split('_')[0]+ '--'+ '_'.join(sorted(set(x), key=x.index)) + '/'

    if 'Demonstration' in s:
      try:
        s = s[:s.index('_')] + '/'
      except ValueError:
       pass
    return s

def jaccard_distance(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    :param bb1: list [x1, x2, y1, y2]
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    :param bb2: list [x1, x2, y1, y2]
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    :return: float in [0, 1]
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou * (iou > 0.5)

def find_best_bbox(box, predicted_boxes):
    """
    Find the corresponding predicted bounding box 
    compared to the ground truth
    :param box: ground truth bounding box
    :param predicted_boxes: list of predicted bounding boxes
    :return: index of the corresponding bbox, jaccard distance
    """
    if type(box) == list:
        box = ' '.join(map(str,box))
    (x1, y1, w, h) = map(int,box.split()[:4])
    boxA = [x1, y1, x1+w, y1+h]
    l = []
    # boxB : [x1, x2, y1, y2] (top-left and bottom-right)
    for boxB in predicted_boxes:
        l.append(jaccard_distance(boxA, boxB))
    if len(l) > 0:
      return np.argmax(l), np.max(l)
    else:
      return -1, 0

def mean_jaccard(truth_boxes, predicted_boxes, only_tp=True, blurred=0):
    """
    Compute the average Jaccard distance for the bounding boxes of 
    one picture. 
    :param truth_boxes: ground truth bounding boxes
    :param predicted_boxes: predicted bounding boxes
    :param only_tp: boolean to only keep true positive bounding bo
    :return: mean jaccard and number of TP (None, if no TP found)
    """
    l = []
    for truth_box in truth_boxes:
        if int(truth_box.split()[4]) >= blurred:
            _, jd = find_best_bbox(truth_box, predicted_boxes)
            l.append(jd)
    if only_tp:
        l = [k for k in l if k > 0]
    if len(l) > 0:
        return np.mean(l), len(l)

def compute_stats(data_dir, truth, predictions, blurred=0):
    """
    Compute the mean Jaccard distance and the ratio of predicted bounding
    boxes compared to the number of actual bounding boxes
    :param data_dir: directory path with the pictures
    :param truth: dict of actual annotations of the bounding boxes 
            d[name] = [(x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose)]
    :param predictions: list of predicted bounding boxes 
            keeping the same order of glob.glob(pictures folder)
    :param blurred: 0 for all faces, 1 for normal blurred faces, 2 for heavy blurred faces
    :return: (len(pictures), 4) numpy array and the corresponding panda DataFrame
            ['mean Jaccard', 'Nb_Truth_Bboxes', 'Nb_Pred_Bboxes', 'Ratio_Bboxes']
    """
    pictures = glob.glob(data_dir + '*')
    n_pictures = len(pictures)
    jaccard, n_truth_boxes, n_pred_boxes = [], [], []
    a = np.zeros((n_pictures,4))
    
    for idx in range(n_pictures):
        truth_boxes = truth[pictures[idx].replace(data_folder, '')]
        temp = mean_jaccard(truth_boxes, predictions[idx], blurred=blurred)
        if temp:
            mean_jac, nb_pred = temp
        else:
            mean_jac, nb_pred = None, 0
        jaccard.append(mean_jac)
        n_truth_boxes.append(len([k for k in truth_boxes if int(k.split()[4]) >= blurred]))
        n_pred_boxes.append(nb_pred)  
    
    a[:,0] = jaccard
    a[:,1] = n_truth_boxes
    a[:,2] = n_pred_boxes
    a[:,3] = a[:,2]/a[:,1]
    df = pd.DataFrame(a, columns=['mJaccard', 'Nb_Truth_Bboxes', 'Nb_Pred_Bboxes', 'Ratio_Bboxes'])
    df['Folder'] = data_dir.replace(data_folder, '')
    return a, df
