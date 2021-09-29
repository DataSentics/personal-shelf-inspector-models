import io
from google.cloud import vision
from google.oauth2 import service_account
from PIL import Image, ImageDraw, ImageFont
import numpy as np


def draw_ocr(detection):
    
    # load image
    image = Image.open(detection[0]).convert('RGB')
    # create an object for drawing
    draw = ImageDraw.Draw(image)
    # draw bboxes and text (first string is summary of all outputs)
    for k, bbox in enumerate(detection[2][1:]):
        tl = bbox[0]
        br = bbox[2]
        draw.rectangle([tl, br], outline=(0,0,255), width=1)
        draw.text((int(np.mean([tl[0],br[0]])), tl[1]-16),
                  detection[1][k+1],
                  font=ImageFont.truetype("DejaVuSans.ttf",
                                          size=15,
                                          encoding="utf-8"),
                  fill=(0,0,255))

    return image


def detect_text(path):
    """Detects text in the file."""

    creds = service_account.Credentials.from_service_account_file('./ocr/key.json')
    client = vision.ImageAnnotatorClient(credentials=creds)
        
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    
    response = client.text_detection(image=image)
    texts = response.text_annotations
    
    det_text = []
    det_bbox = []
    
    for text in texts:
    
        det_text.append(text.description)
        det_bbox.append([(vertex.x, vertex.y) for vertex
                         in text.bounding_poly.vertices])
    
    if response.error.message:
        raise Exception(
            '{}\nFor more info on error messages, check: '
            'https://cloud.google.com/apis/design/errors'.format(
                response.error.message))

    return [path, det_text, det_bbox]


def save_cuts_for_detection(cuts):
    """
    Save list of cuts to pre-selected folder as preparation for detection
    """

    cuts_path = [['./test_images/cut_{:02d}.png'.format(k),
                  Image.fromarray(im).save('./test_images/cut_{:02d}.png'.format(k))]
                 for k, im in enumerate(cuts)]

    return [path[0] for path in cuts_path]
