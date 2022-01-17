import os
import yaml 
import shutil
import random
from PIL import Image, ImageDraw
import tqdm


class AffineTransform:
    def __init__(self, w, h, grid_dim_x, grid_dim_y, target_size, row, col):
        cell_width = target_size / grid_dim_x
        cell_height = target_size / grid_dim_y

        # scale width and height to the cell size
        img_aspect_ratio = w / h
        cell_aspect_ratio = cell_width / cell_height

        if img_aspect_ratio > cell_aspect_ratio:
            # the image is wider than the cell
            # scale the width to the cell size

            self.scale_factor = cell_width / w
        else:
            # the image is taller than the cell
            # scale the height to the cell size
            self.scale_factor = cell_height / h

        self.x_offset = col * cell_width
        self.y_offset = row * cell_height

        # center inside the cell
        self.x_offset += (cell_width - w * self.scale_factor) / 2
        self.y_offset += (cell_height - h * self.scale_factor) / 2


    def forward(self, x, y, w, h):
        return (x * self.scale_factor) + self.x_offset, (y * self.scale_factor) + self.y_offset, w * self.scale_factor, h * self.scale_factor

    def inverse(self, x, y, w, h):
        return (x - self.x_offset) / self.scale_factor, (y - self.y_offset) / self.scale_factor, w / self.scale_factor, h / self.scale_factor

class Collage:
    def __init__(self, grid_dim_x, grid_dim_y, target_size):
        self.target_size = target_size
        self.collage_img = Image.new('RGB', (target_size, target_size))
        self.draw = ImageDraw.Draw(self.collage_img)

        self.grid_dim_x = grid_dim_x
        self.grid_dim_y = grid_dim_y

        self.all_annotations = []
        self.imgs = []
    
    def add_image(self, img, row, col, annotations=[]):
        transformed_img = TransformedImage(img, self.grid_dim_x, self.grid_dim_y, self.target_size, row, col, annotations)
        self.imgs.append(transformed_img)
        self.all_annotations.extend(transformed_img.map_annotations())
        transformed_img.draw_on_collage(self.collage_img)

    def save(self, out_img_dir, out_label_dir, file_prefix):
        self.collage_img.save(os.path.join(out_img_dir, file_prefix + '.jpg'))

        with open(os.path.join(out_label_dir, file_prefix + '.txt'), 'w') as f:
            for annotation in self.all_annotations:
                # convert annotations to YOLO format
                x, y, w, h = annotation['x'], annotation['y'], annotation['w'], annotation['h']
                cx, cy = x + w/2, y + h/2

                # convert to relative coordinates
                cx, cy, w, h = [c / self.target_size for c in (cx, cy, w, h)]

                f.write('{} {} {} {} {}\n'.format(annotation['label'], cx, cy, w, h))


class TransformedImage:
    def __init__(self, img, grid_dim_x, grid_dim_y, target_size, row, col, annotations = []):
        self.orig_img = img
        self._annotations = annotations

        self.transform_fn = AffineTransform(img.width, img.height, grid_dim_x, grid_dim_y, target_size, row, col)
        
    def draw_on_collage(self, collage_img):
        # resize the image
        new_x, new_y, new_w, new_h = self.transform_fn.forward(0, 0, self.orig_img.width, self.orig_img.height)
        new_img = self.orig_img.resize((int(new_w), int(new_h)))

        # paste the image onto the collage
        collage_img.paste(new_img, (int(new_x), int(new_y)))


    def map_annotations(self):
        new_annotations = []
        for annotation in self._annotations:
            new_ann = {}
            new_ann['x'], new_ann['y'], new_ann['w'], new_ann['h'] = self.transform_fn.forward(annotation['x'], annotation['y'], annotation['w'], annotation['h'])
            new_ann['label'] = annotation['label']
            new_annotations.append(new_ann)
        
        return new_annotations



def create_collage_img(src_data, grid_dim_x, grid_dim_y, target_size):
    
    collage = Collage(grid_dim_x, grid_dim_y, target_size)

    for i, (img_file, label_file) in enumerate(src_data):
        
        # load the image
        img = Image.open(img_file)

        # load the annotations
        src_annotations = []

        if label_file is not None:
            with open(label_file, "r") as f:
                for line in f:
                    # class_id, center_x, center_y, width, height
                    line = line.strip().split(" ")
                    label = line[0]
                    center_x, center_y, w, h = [float(x) for x in line[1:]]
    
                    # convert to absolute coordinates - top-left, width, height
                    img_w, img_h = img.size
    
                    ann = {
                        'label': label,
                        'x': (center_x - (w / 2)) * img_w,
                        'y': (center_y - (h / 2)) * img_h,
                        'w': w * img_w,
                        'h': h * img_h
                    }
    
                    src_annotations.append(ann)
        
        row, col = i % grid_dim_y, i // grid_dim_y

        collage.add_image(img, row, col, src_annotations)
            
    return collage


#def show_collage(collage_img, collage_annotations):
#    # show the collage
#    # draw the bounding boxes
#    # annotations are in the form of (label, [center_x, center_y, width, height]) in relative coordinates
#
#    # copy the collage image
#    cp_img = collage_img.copy()
#    draw = ImageDraw.Draw(cp_img)
#
#    for label, coords in collage_annotations:
#        # scale to the collage image size
#        coords = [c * collage_img.size[0] for c in coords]
#        center_x, center_y, width, height = coords
#
#        rect = (center_x - width / 2, center_y - height / 2, center_x + width / 2, center_y + height / 2)
#        print(rect)
#        draw.rectangle(rect, fill="#800080",  outline="red", width=5)
#
#    # show using matplotlib
#    import matplotlib.pyplot as plt
#    plt.imshow(collage_img)
#    plt.show()



def convert_dataset(src_img_dir, src_label_dir, out_img_dir, out_label_dir, grid_dim_x, grid_dim_y, target_size, n_iterations):
    # get all the image and label files
    src_img_files = sorted([f for f in os.listdir(src_img_dir) if f.endswith(".jpg")])
    src_label_files = sorted([f for f in os.listdir(src_label_dir) if f.endswith(".txt")])
    
    # check, that the lists are the same (except for the extension)
    assert len(src_img_files) == len(src_label_files)
    for i in range(len(src_img_files)):
        assert src_img_files[i].split(".")[0] == src_label_files[i].split(".")[0]

    # add the path prefix
    src_img_files = [os.path.join(src_img_dir, f) for f in src_img_files]
    src_label_files = [os.path.join(src_label_dir, f) for f in src_label_files]

    src_data = list(zip(src_img_files, src_label_files)) * n_iterations

    random.shuffle(src_data)

    # create collages of grid_size x grid_size images
    # the collage is a square image, so the target size is the side length

    for i in tqdm.tqdm(range(0, len(src_data), grid_dim_x * grid_dim_y)):
        # create a collage

        collage = create_collage_img(src_data[i:i+min(len(src_data), grid_dim_x * grid_dim_y)], grid_dim_x, grid_dim_y, target_size)

        # save the collage
        collage.save(out_img_dir, out_label_dir, file_prefix = f"collage_{i}")

def detect_dataset(src_img_dir, out_img_dir, weights, grid_dim_x, grid_dim_y, target_size):
    # TODO

    # load the YOLOv5s model
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    print("half: {}".format(half))
    if half:
        model.half()  # to FP16
    # get all the image files
    src_img_files = sorted([f for f in os.listdir(src_img_dir) if f.endswith(".jpg")])

    # create collages of grid_size x grid_size images

    for i in tqdm.tqdm(range(0, len(src_img_files), grid_dim_x * grid_dim_y)):
        # create a collage
        collage = create_collage_img([(os.path.join(src_img_dir, f), None) for f in src_img_files[i:i+min(len(src_img_files), grid_dim_x * grid_dim_y)]], grid_dim_x, grid_dim_y, target_size)



        # save the collage
        collage.save(out_img_dir, None, file_prefix = f"collage_{i}")

        # detect the collage
        collage.detect(weights)

        # save the collage
        collage.save(out_img_dir, None, file_prefix = f"collage_{i}")
    


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the YOLO dataset")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--target_size", type=int, required=True, help="Size of the target square image.")
    parser.add_argument("--grid_x", type=int, required=True, help="Number of grid columns.")
    parser.add_argument("--grid_y", type=int, required=True, help="Number of grid rows.")
    parser.add_argument("--n_iterations", type=int, default=1, help="Number of random passes over the dataset.")

    parser.add_argument("--detect", action="store_true", help="Detect the dataset and create the output with cut-out images.")
    parser.add_argument("--weights", type=str, default="./temp/best.pt", help="Only used if --detect is set. Path to the model to use for detection.")


    args = parser.parse_args()

    # create output directory
    # delete it, if it already exists
    if os.path.exists(args.output_path):
        print(f"Deleting existing output directory: {args.output_path}")
        shutil.rmtree(args.output_path)
    os.makedirs(args.output_path)

    if args.detect:
        print("Running inference...")
        # detect the dataset
        detect_dataset(args.dataset_path, args.output_path, args.weights, args.grid_x, args.grid_y, args.target_size)
        sys.exit(0)

    # create the settings.yaml file
    settings = {
        "train": os.path.join(args.output_path, "images/train"),
        "val": os.path.join(args.output_path, "images/val"),
        "nc": 3,
        "names": ["price", "price_dec", "product_name"],
    }

    with open(os.path.join(args.output_path, "settings.yaml"), "w") as f:
        yaml.dump(settings, f)
    
    for mode in ['train', 'val']:
        src_img_dir = os.path.join(args.dataset_path, "images", mode)
        src_label_dir = os.path.join(args.dataset_path, "labels", mode)

        out_img_dir = os.path.join(args.output_path, "images", mode)
        out_label_dir = os.path.join(args.output_path, "labels", mode)

        os.makedirs(out_img_dir)
        os.makedirs(out_label_dir)

        convert_dataset(src_img_dir, src_label_dir, out_img_dir, out_label_dir, args.grid_x, args.grid_y, args.target_size, args.n_iterations)

