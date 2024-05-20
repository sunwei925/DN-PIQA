import torch
import argparse
import pandas as pd
import os
import warnings
import sys

from PIL import Image
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import detect_face

def main(args):
    warnings.filterwarnings("ignore")


    image_dir = args.image_dir
    face_save_dir = args.face_save_dir
    csv_path = args.csv_path
    data_df = pd.read_csv(csv_path)

    n_image = len(data_df)

    if not os.path.exists(face_save_dir):
        os.makedirs(face_save_dir)

    for i in range(0,n_image):
        image_name = data_df['IMAGE'][i]
        image_name_str = image_name.split('.')[0]

        image_path = os.path.join(image_dir, image_name)

        image = Image.open(image_path)
        image = image.convert('RGB')

        boxes = detect_face.detect(image_path)

        frame_draw = image.copy()

        try:
            face_image = frame_draw.crop([boxes[0][0]-20, boxes[0][1]-20, boxes[0][2]+20, boxes[0][3]+20])
        except TypeError:
            print('fail to detect face image of ' + image_name)
            face_image = frame_draw
        face_image.save(os.path.join(face_save_dir, image_name_str + '.png'))

def parse_args():
    """Parse input arguments. """
    parser = argparse.ArgumentParser(description="Face image extraction")
    parser.add_argument('--csv_path', type=str)
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--face_save_dir', type=str)


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    main(args)