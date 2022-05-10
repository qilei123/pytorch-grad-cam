import argparse
import cv2
import numpy as np
import torch
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

import os
import glob

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument(
        '--image-path',
        type=str,
        default='./examples/both.png',
        help='Input image path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='scorecam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    parser.add_argument(
        '--pth_dir',
        type=str)
    
    parser.add_argument(
        '--save_dir',
        type=str)

    parser.add_argument(
        '--model_name',
        type=str)

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')

    return args


def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def main1():
    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    model = timm.create_model(args.model_name, num_classes=2,
                            in_chans=3, pretrained=False, checkpoint_path=args.pth_dir)
    model.eval()

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    
    if args.use_cuda:
        model = model.cuda()

    target_layers = [model.layers[-1].blocks[-1].norm2]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    cam = methods[args.method](model=model,
                               target_layers=target_layers,
                               use_cuda=args.use_cuda,
                               reshape_transform=reshape_transform)
    img_paths = glob.glob(os.path.join(args.image_path,"*.jpg"))


    for img_path in img_paths:
        rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255
        
        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        '''
        img = Image.open(args.image_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0) # transform and add batch dimension
        input_tensor = input_tensor.cuda()
        '''
        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested category.
        targets = None

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32

        grayscale_cam = cam(input_tensor=input_tensor,
                            targets=targets,
                            eigen_smooth=args.eigen_smooth,
                            aug_smooth=args.aug_smooth)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam)
        cv2.imwrite(os.path.join(args.save_dir,os.path.basename(img_path)), cam_image)

def main2():
    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")

    model = timm.create_model(args.model_name, num_classes=2,
                            in_chans=3, pretrained=False, checkpoint_path=args.pth_dir)
    model.eval()

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    
    if args.use_cuda:
        model = model.cuda()

    target_layers = [model.layers[-1].blocks[-1].norm2]

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    cam = methods[args.method](model=model,
                               target_layers=target_layers,
                               use_cuda=args.use_cuda,
                               reshape_transform=reshape_transform)
    img_paths = glob.glob(os.path.join(args.image_path,"*.jpg"))

    img_record = open(os.path.join(args.image_path,"test.txt"))

    record = img_record.readline()

    while record:

        img_path = os.path.join("test",record[:-3])
        gt_label = record[-2]

        rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
        rgb_img = cv2.resize(rgb_img, (224, 224))
        rgb_img = np.float32(rgb_img) / 255
        
        input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        '''
        img = Image.open(args.image_path).convert('RGB')
        input_tensor = transform(img).unsqueeze(0) # transform and add batch dimension
        input_tensor = input_tensor.cuda()
        '''
        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested category.
        targets = None

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        cam.batch_size = 32

        grayscale_cam,outputs = cam(input_tensor=input_tensor,
                            targets=targets,
                            eigen_smooth=args.eigen_smooth,
                            aug_smooth=args.aug_smooth)

        print(outputs)

        # Here grayscale_cam has only one image in the batch
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(rgb_img, grayscale_cam)

        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        cv2.imwrite(os.path.join(args.save_dir,os.path.basename(img_path)), cam_image)

        record = img_record.readline()

if __name__ == '__main__':
    """ python swinT_example.py -image-path <path_to_image>
    Example usage of using cam-methods on a SwinTransformers network.

    """
    main2()

