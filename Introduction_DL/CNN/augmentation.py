import torch
from torchvision.transforms import functional, RandomResizedCrop, ColorJitter, RandomErasing
import random
import numpy as np
import matplotlib.pyplot as plt


def rotate_image(image):
    angle = np.random.randint(-30, 30)
    return functional.rotate(image, angle)


def blur_image(image):
    sigma = np.random.uniform(0.5, 1.5)
    return functional.gaussian_blur(image, kernel_size=[3, 3], sigma=sigma)


def flip(image):
    # return functional.hflip(image) if np.random.rand() < 0.5 else image
    return functional.hflip(image)


def color_jitter(image):
    transform = ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
    return transform(image)


def random_affine(image):
    return functional.affine(image,
                             angle=np.random.randint(-15, 15),
                             translate=(0.05, 0.05),
                             scale=np.random.uniform(0.9, 1.1),
                             shear=np.random.randint(-5, 5))


def random_erasing(image):
    transform = RandomErasing(p=0.5)
    return transform(image)


def add_noise(image):
    noise = torch.randn(image.size()) * 0.1
    return image + noise.clamp(0, 1)


def random_grayscale(image):
    return functional.rgb_to_grayscale(image, num_output_channels=3) if np.random.rand() < 0.3 else image


def apply_random_filters(image, probabilities=[0.3, 0.3, 0.2, 0.2],
                         print_filters: bool = False):
    augmentations = [rotate_image,
                     blur_image,
                     flip,
                     color_jitter,
                     random_affine,
                     random_erasing,
                     add_noise,
                     random_grayscale]


    num_filters = np.random.choice(range(len(probabilities)), p=probabilities)

    # Randomly select and apply the filters
    for _ in range(num_filters):
        func = random.choice(augmentations)
        image = func(image)
        if print_filters:
            print(func.__name__())

    return image


def show_random(data, lables, denorm: bool = True,
            probabilities=[0.3, 0.3, 0.2, 0.2], print_filters: bool = False):
    
    def preprocess(image):
        if denorm:
            image = (image + 1) / 2 * 255
        else:
            image *= 255
        image = image.int()
        image = image.numpy().transpose(1, 2, 0)
        return image

    _, axes = plt.subplots(1, 2)
    ri = np.random.randint(0, len(data))
    print(lables[ri].item())

    image = data[ri]
    
    axes[0].imshow(preprocess(image))
    axes[0].set_title('Image default')

    axes[1].imshow(preprocess(
                    apply_random_filters(
                        image, probabilities, print_filters)))
    axes[1].set_title('Image transformed')

    plt.show()




