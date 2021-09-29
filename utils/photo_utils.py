from PIL import Image as PILImage
from PIL.ExifTags import TAGS
import io
import numpy as np


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


def get_photo(btn_upload, preview=False):
    """Get photo from button message and resize it if desired. """
    im_file = PILImage.open(io.BytesIO(btn_upload.data[-1]))
    im_file = exif_rotate(im_file)
    # rescale 
    if preview:
        # display original photo, resize it for faster display
        basewidth = 400
        wpercent = (basewidth/float(im_file.size[0]))
        hsize = int((float(im_file.size[1])*float(wpercent)))
        im_file = im_file.resize((basewidth,hsize), PILImage.ANTIALIAS)    
    return im_file


def get_centermost_detections(im,
                              det,
                              n=1,
                              mode="cut",
                              bigger_box_by_pct=0.01):
    """ 
    Get n detections closest to image center. Returns either coordinates or cuts.
    """
    # get center of image
    im_file = np.array(im)
    img_center = np.array(im_file.shape[0:2])/2

    # calculate distances from center of pricetags to center of image
    distances = [
        np.linalg.norm(
            np.array(
                [
                    np.mean(r[[0,2]]),
                    np.mean(r[[1,3]])
                ]
            )-img_center
        )
        for r in det["rois"]
    ]
    
    # select n pricetags closest to the center
    # for now, n=1
    pricetags_indices = np.argsort(distances)[0:min(n,len(det["cls"]))]
    
    out = []
    for ind_pricetag in pricetags_indices:
        # cut pricetag from image
        bbox = det["rois"][ind_pricetag]
        img_size = im_file.shape

        #use a bit wider bbox
        ratio = bigger_box_by_pct
        bbox_new = bbox
        bbox_new[0] = int(max(0,bbox[0]-ratio*(bbox[2]-bbox[0])))
        bbox_new[1] = int(max(0,bbox[1]-ratio*(bbox[3]-bbox[1])))
        bbox_new[2] = int(min(img_size[0],bbox[2]+ratio*(bbox[2]-bbox[0])))
        bbox_new[3] = int(min(img_size[1],bbox[3]+ratio*(bbox[3]-bbox[1])))

        if mode == "return":
            out.append([
                bbox_new[0],
                bbox_new[1],
                bbox_new[2],
                bbox_new[3]
            ])
        elif mode == "cut":
            out.append(               
                im_file[
                    int(bbox_new[0]) : int(bbox_new[2]), 
                    int(bbox_new[1]) : int(bbox_new[3]),
                    :
                ]
              )

    return out
