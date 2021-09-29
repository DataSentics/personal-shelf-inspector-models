import numpy as np
from copy import deepcopy


def google_coords_to_yolo_coords(google_detections):
    # input of type e.g.
    # [[(59, 44), (477, 44), (477, 285), (59, 285)],
    #   [(59, 45), (156, 45), (156, 62), (59, 62)]]
    # output : array([[44,59,285,477], [45,59,62,156]])
    one_box_coords = np.array(google_detections)
    maxs = np.max(google_detections, axis=1)
    mins = np.min(google_detections, axis=1)
    # to one array
    coords = np.concatenate((mins,maxs),axis=1)
    # swap columns so that 0 and 3 are row indices
    coords = coords[:,[1,0,3,2]]
    return coords

def intersect_area(a, b):
    dx = min(a[3], b[3]) - max(a[1], b[1])
    dy = min(a[2], b[2]) - max(a[0], b[0])
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    else:
        return 0
    
def most_overlapping_bbox_index(one_bbox, bbox_candidates, n_to_return=1):
    intersect_areas = np.apply_along_axis(
        intersect_area,
        1,
        bbox_candidates,
        b=one_bbox
    )
    #print(intersect_areas)
    return (-intersect_areas).argsort()[:n_to_return]

def get_overlaps_yolo_google(google_detections,
                             yolo_detections,
                             yolo_classes,
                             what_to_find,
                             what_to_ignore=None,
                             n_to_return=1):
    """
    Looks for the google detection with highest overlap with the yolo detection and returns text and bbox.

    :param google_detections:  list [image_name, [text], [coords]], detections from google API
    :param yolo_detections: detections from Yolo model for price and product name
    :what_to_find: str, a class name to look for (will be asserted)
    :what_to_ignore: str, a class name to ignore in detection
    :n_to_return: int, number of matches to return
    """
    yolo_det = deepcopy(yolo_detections)
    # TODO: what_to_ignore: masks currently only first occurence in detections
    assert what_to_find in yolo_classes, "Choose what_to_find contained in 'yolo_classes'"

    ## transform google api detection bboxes to yolo format (which is based on mask rcnn format here)
    google_bboxes = google_coords_to_yolo_coords(google_detections[2]) 
    # find desired object between yolo classes
    detection_class_id = yolo_classes.index(what_to_find)
    detection_id = list(yolo_det["cls"]).index(detection_class_id)
    detection_bbox = yolo_det["rois"][detection_id,:]
    
    if what_to_ignore:
        # ignore intersection with what_to_ignore, set its coords to some impossible value to ensure 0 intersection
        ignore_class_id = yolo_classes.index(what_to_ignore)
        yolo_det["rois"][detection_id,:] = [-100,-100,-100,-100]
    #print(detection_bbox)
    # find google detection with highest overlap, get rid of first bbox, it probably contains convex hull of all detections
    result_index = most_overlapping_bbox_index(detection_bbox,google_bboxes[1:,:], n_to_return=n_to_return) + 1

    return np.array(google_detections[1])[result_index], np.array(google_bboxes)[result_index,:]