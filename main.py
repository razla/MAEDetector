from torchvision.models import resnet50, ResNet50_Weights

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import AdversarialPatch

import matplotlib.pyplot as plt

from PIL import Image

import numpy as np

import requests

import torch


from utils import prepare_model
from utils import imagenet_mean
from utils import imagenet_std
from utils import show_image
from utils import run_one_image
from utils import load_dataset


def load_image(name):
    # load an image
    img_url = 'https://user-images.githubusercontent.com/11435359/147738734-196fd92f-9260-48d5-ba7e-bf103d29364d.jpg' # fox, from ILSVRC2012_val_00046145
    # img_url = 'https://user-images.githubusercontent.com/11435359/147743081-0428eecf-89e5-4e07-8da5-a30fd73cc0ba.jpg' # cucumber, from ILSVRC2012_val_00047851
    img = Image.open(requests.get(img_url, stream=True).raw)
    img = img.resize((224, 224))
    img = np.array(img) / 255.

    assert img.shape == (224, 224, 3)

    # normalize by ImageNet mean and std
    img = img - imagenet_mean
    img = img / imagenet_std

    plt.rcParams['figure.figsize'] = [5, 5]
    show_image(torch.tensor(img), name=name)

    return img

if __name__ == '__main__':

    target_name = 'toaster'
    image_shape = (224, 224, 3)
    clip_values = (0, 1)
    nb_classes = 1000
    batch_size = 16
    scale_min = 0.4
    scale_max = 1.0
    rotation_max = 22.5
    learning_rate = 5000.
    max_iter = 500

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img = load_image('orig.png')

    chkpt_dir = 'mae_visualize_vit_large.pth'
    model_mae = prepare_model(chkpt_dir, 'mae_vit_large_patch16')
    print('Model loaded.')

    torch.manual_seed(2)
    print('MAE with pixel reconstruction:')
    run_one_image(img, model_mae, 'recon.png')

    classifier = resnet50(weights=ResNet50_Weights.DEFAULT).eval().to(device)
    art_classifier = PyTorchClassifier(model=classifier,
                                       loss=torch.nn.CrossEntropyLoss(),
                                       input_shape=image_shape,
                                       nb_classes=nb_classes,
                                       clip_values=clip_values)

    ap = AdversarialPatch(classifier=art_classifier, rotation_max=rotation_max, scale_min=scale_min, scale_max=scale_max, patch_shape=(3, 20, 20), learning_rate=learning_rate, max_iter=max_iter, batch_size=batch_size)

    (x_test, y_test), min_pixel_value, max_pixel_value = load_dataset()

    adv = ap.generate(x_test[:50])

    for i, (x, y) in enumerate(zip(x_test[:10], y_test[:10])):
        x = x.unsqueeze(dim=0).to(device)
        y = y.to(device)