import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import cm


def read_annotations(path):
    # each line is a list of [label, x1, y1, width, height]
    # coordinates are in the 0-1 scale
    with open(path, 'r') as f:
        lines = f.readlines()
    bboxes = []
    labels = []
    for line in lines:
        line = line.strip().split()
        label = line[0]
        x1 = float(line[1])
        y1 = float(line[2])
        x2 = float(line[3])
        y2 = float(line[4])
        bboxes.append([x1, y1, x2, y2])
        labels.append(label)
    return bboxes, labels



def visualize_bboxes_on_yolo_image(image_path, annotations_path=None):
    """
    Visualize bounding boxes on the image.
    :param image_path: path to the image
    :param annotations_path: path to the annotations in the YOLOv5 format

    The dataset directory structure is as follows:

        dataset/
            images/
                train/
                    0000.jpg
                    0001.jpg
                    ...
                val/
                    0000.jpg
                    0001.jpg
                    ...
            labels/
                train/
                    0000.txt
                    0001.txt
                    ...
                val/
                    0000.txt
                    0001.txt
                    ...
    """


    if annotations_path is None:
        img_name = os.path.basename(image_path)
        split_path = os.path.basename(os.path.dirname(image_path))

        img_prefix = ".".join(img_name.split('.')[:-1])
        annotations_path = os.path.join(os.path.dirname(image_path), '../..',  'labels', split_path, img_prefix + '.txt')

    bboxes, labels = read_annotations(annotations_path)

    visualize_bboxes(image_path, bboxes, labels)
        



def visualize_bboxes(img_path, bboxes, labels=None, label_map=None, num_classes=None):
    """
    Show image with bounding boxes
    :param img: image
    :param bboxes: bounding boxes
    :param labels: labels
    :param colors: colors
    :param classes: classes
    :return:
    """
    if labels is None:
        labels = [0] * len(bboxes)

    if label_map is None:
        labels = list(map(str, labels))
    else:
        labels = list(map(label_map.get, labels))

    if num_classes is None:
        num_classes = len(set(labels))

    img = cv2.imread(img_path)


    # generate color spectrum
    colors = cm.rainbow(np.linspace(0, 1, num_classes))
    # convert to 0-255 scale
    colors = (colors * 255).astype(np.uint8).tolist()



    label_set = set(labels)
    color_map = {label: colors[i] for i, label in enumerate(label_set)}

    img = img.copy()
    for i in range(len(bboxes)):
        center_x, center_y, w, h = bboxes[i]

        # bbox is (centex_x, centex_y, w, h) in the 0-1 scale
        # convert to top-left and bottom-right in the image scale
        x1 = int((center_x - w/2) * img.shape[1])
        y1 = int((center_y - h/2) * img.shape[0])
        x2 = int((center_x + w/2) * img.shape[1])
        y2 = int((center_y + h/2) * img.shape[0])

        label = labels[i]
        color = color_map[label]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    plt.imshow(img[:, :, ::-1])
    plt.show()


if __name__ == '__main__':
    import argparse
    import sys

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', type=str, default=None, help='path to the dataset')
    parser.add_argument('--image_path', '-i', type=str, default=None,
                        help='path to the image')
    parser.add_argument('--annotations_path', '-a', type=str, default=None,
                        help='path to the annotations in the YOLOv5 format')

    args = parser.parse_args()

    if args.dataset_path is not None:
        for img_path in os.listdir(args.dataset_path):
            visualize_bboxes_on_yolo_image(os.path.join(args.dataset_path, img_path))
    elif args.image_path is not None:
        if args.annotations_path is None:
            print("Please specify the annotations path")
            sys.exit(1)
        else:
            annotations_path = args.annotations_path
        visualize_bboxes(args.image_path, args.annotations_path)
    else:
        raise ValueError('Either dataset_path or image_path should be specified')



