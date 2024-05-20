import os
import sys
from PIL import Image
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))


import detect_face
import LIQE

import DN_PIQA

import torch
from torchvision import transforms

def main(args):
    
    image_path = args.image_path
    image_name = args.image_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define and load Face IQA model
    model = DN_PIQA.PIQ_model(pretrained_path = None, pretrained_path_face= None)
    model.load_state_dict(torch.load(r'./weights/PIQ_model.pth'))
    model = model.to(device)

    model_LIQE = LIQE.LIQE_feature()
    model_LIQE = model_LIQE.to(device)

    image_transforms = transforms.Compose([transforms.Resize(448),transforms.FiveCrop(384), \
        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])), \
            (lambda crops: torch.stack([transforms.Normalize(mean=[0.485, 0.456, 0.406], \
                std=[0.229, 0.224, 0.225])(crop) for crop in crops]))])


    print('Calculate the quality socre of image: ' + image_name)

    # the overall image
    overall_image = Image.open(os.path.join(image_path, image_name))

    # copy the overall image for calculating the LIQE features
    overall_image_LIQE = overall_image.copy()

    overall_image = overall_image.convert('RGB')

    # face image detection
    print('\n')
    print('Display the message of detecting face')
    boxes = detect_face.detect(os.path.join(image_path, image_name))
    overall_image_for_face = overall_image.copy()
    try:
        face_image = overall_image_for_face.crop([boxes[0][0]-20, boxes[0][1]-20, boxes[0][2]+20, boxes[0][3]+20])
    except TypeError:
        print('fail to detect face image: ' + image_name)
        face_image = overall_image_for_face
    
    # LIQE feature extraction
    overall_image_LIQE = transforms.ToTensor()(overall_image_LIQE).unsqueeze(0)
    LIQE_feature = model_LIQE(overall_image_LIQE)

    # image transforms
    overall_image = image_transforms(overall_image)
    overall_image = overall_image.unsqueeze(0)
    face_image = image_transforms(face_image)
    face_image = face_image.unsqueeze(0)

    
    with torch.no_grad():
        model.eval()
        # calculate the image scores using the five crops manner
        bs, ncrops, c, h, w = overall_image.size()
        overall_image = overall_image.to(device)
        face_image = face_image.to(device)
        LIQE_feature = torch.cat([LIQE_feature, LIQE_feature, LIQE_feature, LIQE_feature, LIQE_feature], dim=0)
        LIQE_feature = LIQE_feature.to(device)

        outputs = model(overall_image.view(-1, c, h, w), face_image.view(-1, c, h, w),LIQE_feature)
        image_score = outputs.view(bs, ncrops, -1).mean(1).item()
    
    print('\n')
    print('The quality score of image {} is {:.4f}'.format(image_name, image_score))

def parse_args():
    """Parse input arguments. """
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--image_name', type=str)


    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    main(args)