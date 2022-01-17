import os
import json
import shutil
import argparse
import random
import numpy as np
import traceback
from PIL import Image
from copy import deepcopy
from PIL.ExifTags import TAGS

from bbox_visualize import visualize_bboxes_on_yolo_image


def via_to_yolo(via_ann, path_in, path_out, mode="train"):
    os.mkdir(path_out + f"labels/{mode}")
    os.mkdir(path_out + f"images/{mode}")
    n_ann = len(via_ann.keys())
    cnt = 1

    already_done = set()


    for i, ann in via_ann.items():
        # print(ann)
        # print(f"{cnt}/{str(n_ann)}")
        img_name = ann["filename"]
        if img_name in already_done:
            print("Duplicate image:", img_name, i)
            continue
        already_done.add(img_name)

        # remove the extension
        img_name = img_name.split(".")[0]

        try:
            im = Image.open(os.path.join(path_in, ann["filename"]))
        except FileNotFoundError:
            print(f"{ann['filename']} not found, skipping")
            continue

        # exif rotate to prevent wrong coords calculation
        im = exif_rotate(im)
        width = im.size[0]
        height = im.size[1]

        labels_path = os.path.join(path_out, f"labels/{mode}/{img_name}.txt")
        try:
            with open(labels_path, "a") as file:
                for reg in ann["regions"]:
                    label = reg["region_attributes"]["label"]
                    class_id = class_to_id.get(label)
                    ann_type = reg["shape_attributes"]["name"]
                    if ann_type == "polygon":
                        # x==columns, y==rows, zero in top-left corner
                        x_cnt = (
                            np.min(reg["shape_attributes"]["all_points_x"])
                            + (
                                np.max(reg["shape_attributes"]["all_points_x"])
                                - np.min(reg["shape_attributes"]["all_points_x"])
                            )
                            / 2
                        )
                        y_cnt = (
                            np.min(reg["shape_attributes"]["all_points_y"])
                            + (
                                np.max(reg["shape_attributes"]["all_points_y"])
                                - np.min(reg["shape_attributes"]["all_points_y"])
                            )
                            / 2
                        )
                        w = np.max(reg["shape_attributes"]["all_points_x"]) - np.min(
                            reg["shape_attributes"]["all_points_x"]
                        )
                        h = np.max(reg["shape_attributes"]["all_points_y"]) - np.min(
                            reg["shape_attributes"]["all_points_y"]
                        )
                    elif ann_type == "rect":
                        x_cnt = (
                            reg["shape_attributes"]["x"]
                            + reg["shape_attributes"]["width"] / 2
                        )
                        y_cnt = (
                            reg["shape_attributes"]["y"]
                            + reg["shape_attributes"]["height"] / 2
                        )
                        w = reg["shape_attributes"]["width"]
                        h = reg["shape_attributes"]["height"]

                    x_cnt = x_cnt / width
                    y_cnt = y_cnt / height
                    w = w / width
                    h = h / height

                    if max([x_cnt, y_cnt, w, h]) > 1:
                        print(f"{ann['filename']} has wrong coords - {(x_cnt, y_cnt, w, h)}, skipping")
                        raise Exception("Coordinates >1 at", img_name)
                    file.write(f"{class_id} {x_cnt} {y_cnt} {w} {h}\n")
            output_path = os.path.join(path_out, f"images/{mode}/{img_name}.jpg")
            im.save(output_path)
            # visualize_bboxes_on_yolo_image(output_path)

            cnt += 1

        # except ZeroDivisionError as e:
        except Exception as e:
            #traceback.print_exc()
            print("Unable to process file", ann["filename"], ":", e)
            # remove annotation file to incomplete photo
            try:
                os.remove(labels_path)
            except OSError:
                pass
    return


def exif_rotate(image):
    info = image._getexif()
    if info is not None:
        for tag, value in info.items():
            key = TAGS.get(tag)
            if key == "Orientation":
                orientation = value
                if orientation == 3:
                    image = image.rotate(180, expand=True)
                elif orientation == 6:
                    image = image.rotate(270, expand=True)
                elif orientation == 8:
                    image = image.rotate(90, expand=True)
    return image

def remove_duplicates(via_ann):
    """
    Removes duplicit images from annotations
    """
    duplicates = set()
    image_set = set()

    for i, ann in via_ann.items():
        if ann["filename"] in image_set:
            duplicates.add(ann["filename"])
            # print("Duplicate image:", ann["filename"], i)
        else:
            image_set.add(ann["filename"])

    print("Removed", len(duplicates), "duplicates")
    clean_ann = {ann_id : ann for ann_id, ann in via_ann.items() if ann["filename"] not in duplicates}
    return clean_ann

def get_classes(via_ann):
    all_classes = []
    for i, ann in via_ann.items():
        for reg in ann["regions"]:
            all_classes.append(reg["region_attributes"]["label"])
    return sorted(list(set(all_classes)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path")
    parser.add_argument("--output_path")
    parser.add_argument("--annotations_filename", default="annotations.json")
    parser.add_argument("--train_val_split_ratio", type=float, default=0.2)
    args = parser.parse_args()

    try:
        shutil.rmtree(args.output_path + "images/")
    except:
        pass
    try:
        shutil.rmtree(args.output_path + "labels/")
    except:
        pass

    os.makedirs(args.output_path + "images/")
    os.makedirs(args.output_path + "labels/")

    with open(args.input_path + args.annotations_filename, "r") as f:
        via_data = json.load(f)

    via_data = remove_duplicates(via_data)

    all_classes =  get_classes(via_data)

    class_to_id = {cl: i for i, cl in enumerate(all_classes)}
    print(len(all_classes), " classes")
    print("---------------")
    print(all_classes)
    print("---------------")
    print(class_to_id)

    ## train validation split
    keys = list(via_data.keys())
    keys.sort()


    val_num = int(np.ceil(len(keys) * args.train_val_split_ratio))
    random.seed(10)
    random.shuffle(keys)
    # split
    val_set = keys[:val_num]
    train_set = keys[val_num:]
    # create annotations
    via_orig = deepcopy(via_data)
    via_train = dict((k, via_orig[k]) for k in via_orig.keys() if k in train_set)
    via_val = dict((k, via_orig[k]) for k in via_orig.keys() if k in val_set)

    # convert train dataset
    via_to_yolo(
        via_train,
        args.input_path,
        args.output_path,
        mode="train",
    )

    via_to_yolo(
        via_val,
        args.input_path,
        args.output_path,
        mode="val") 

