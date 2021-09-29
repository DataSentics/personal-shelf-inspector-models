import os
import json
import shutil
import argparse
import random
import numpy as np
from PIL import Image
from copy import deepcopy
from PIL.ExifTags import TAGS


def via_to_coco(via_ann, path_in, path_out, mode="train", ann_type="polygon"):
    """
    Does something that Petr Michal will describe

    :param via_ann: doplni Petr Michal
    :param path_in: ditto
    ...
    ...

    """
    assert ann_type in ["polygon", "rectangle"]
    print("Ann type:" + ann_type)
    os.mkdir(path_out + f"labels/{mode}")
    os.mkdir(path_out + f"images/{mode}")
    n_ann = len(via_ann.keys())
    cnt = 1
    for i, ann in via_ann.items():
        try:
            print(f"{cnt}/{str(n_ann)}")
            img_name = ann["filename"].split(".jpg")[0]
            img_name = img_name.replace(" ", "_")
            im = Image.open(path_in + ann["filename"])
            # exif rotate to prevent wrong coords calculation
            im = exif_rotate(im)
            width = im.size[0]
            height = im.size[1]
            with open(path_out + f"labels/{mode}/" + img_name + ".txt", "w") as file:
                for reg in ann["regions"]:
                    try:
                        label = reg["region_attributes"]["object"]
                    except KeyError:
                        label = reg["region_attributes"]["Object"]
                    class_id = class_to_id.get(label)
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
                        x_cnt = x_cnt / width
                        y_cnt = y_cnt / height
                        w = w / width
                        h = h / height
                    elif ann_type == "rectangle":
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
                        raise Exception("Coordinates >1 at", img_name)
                    file.write(f"{class_id} {x_cnt} {y_cnt} {w} {h}\n")
            im.save(path_out + f"images/{mode}/" + img_name + ".jpg")
        except Exception as e:
            print("Unable to process file", ann["filename"], ":", e)
            # remove annotation file to incomplete photo
            try:
                os.remove(path_out + f"labels/{mode}/" + img_name + ".txt")
            except OSError:
                pass
        cnt += 1
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path_train")
    parser.add_argument("--input_path_val", default=None)
    parser.add_argument("--output_path")
    parser.add_argument("--ann_type", default="polygon")
    parser.add_argument("--annotations_filename", default="annotations.json")
    # applied only if no validation set given
    parser.add_argument("--train_val_split_ratio", default=0.2)
    args = parser.parse_args()

    # input_path = "../orig_fridge_data/"
    # output_path = "../fridge/"

    ## TODO: Try if parent folder exists

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

    with open(args.input_path_train + args.annotations_filename, "r") as f:
        via_train = json.load(f)

    if args.input_path_val is not None:
        with open(args.input_path_val + args.annotations_filename, "r") as f:
            via_val = json.load(f)

    # find all classes

    all_classes_train = []
    for i, ann in via_train.items():
        for reg in ann["regions"]:
            try:
                all_classes_train += [reg["region_attributes"]["object"]]
            except KeyError:
                try: 
                    all_classes_train += [reg["region_attributes"]["Object"]]
                except KeyError: 
                    all_classes_train += [reg["region_attributes"]["label"]]
    all_classes_train = sorted(list(set(all_classes_train)))

    all_classes_val = []
    if args.input_path_val is not None:
        for i, ann in via_val.items():
            for reg in ann["regions"]:
                try:
                    all_classes_val += [reg["region_attributes"]["object"]]
                except KeyError:
                    try: 
                        all_classes_val += [reg["region_attributes"]["Object"]]
                    except KeyError: 
                        all_classes_val += [reg["region_attributes"]["label"]]


        all_classes_val = sorted(list(set(all_classes_val)))

    if len(all_classes_train) < len(all_classes_val):
        print(all_classes_val)
        raise ValueError("Mode classes in val than in train data.")

    all_classes = sorted(list(set(all_classes_train)))
    class_to_id = {cl: i for i, cl in enumerate(all_classes)}
    print(len(all_classes), " classes")
    print("---------------")
    print(all_classes)
    print("---------------")
    print(class_to_id)

    ## train validation split if no validation data provided
    if args.input_path_val is None:
        keys = list(via_train.keys())
        val_num = int(np.ceil(len(keys) * args.train_val_split_ratio))
        random.seed(10)
        random.shuffle(keys)
        # split
        val_set = keys[:val_num]
        train_set = keys[val_num:]
        # create annotations
        via_orig = deepcopy(via_train)
        via_train = dict((k, via_orig[k]) for k in via_orig.keys() if k in train_set)
        via_val = dict((k, via_orig[k]) for k in via_orig.keys() if k in val_set)

    ### new way
    via_to_coco(
        via_train,
        args.input_path_train,
        args.output_path,
        mode="train",
        ann_type=args.ann_type,
    )
    if args.input_path_val is None:
        # copy images from train directory if no val is given (annotations solved before)
        via_to_coco(
            via_val,
            args.input_path_train,
            args.output_path,
            mode="val",
            ann_type=args.ann_type,
        )
    else:
        via_to_coco(
            via_val,
            args.input_path_val,
            args.output_path,
            mode="val",
            ann_type=args.ann_type,
        )
