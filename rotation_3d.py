import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode

def rotation_3d(X, axis, theta, expand=False, fill=0.0):
    """
    The rotation is based on torchvision.transforms.functional.rotate, which is originally made for a 2d image rotation
    :param X: the data that should be rotated, a torch.tensor or an ndarray, with lenx * leny * lenz shape.
    :param axis: the rotation axis based on the keynote request. 0 for x axis, 1 for y axis, and 2 for z axis.
    :param expand:  (bool, optional) – Optional expansion flag. If true, expands the output image to make it large enough to hold the entire rotated image. If false or omitted, make the output image the same size as the input image. Note that the expand flag assumes rotation around the center and no translation.
    :param fill:  (sequence or number, optional) –Pixel fill value for the area outside the transformed image. If given a number, the value is used for all bands respectively.
    :param theta: the rotation angle, Counter-clockwise rotation, [-180, 180] degrees.
    :return: rotated tensor.
    """
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if type(X) is np.ndarray:
        X = torch.from_numpy(X)
        X = X.float()

    X = X.to(device)

    if axis == 0:
        X = rotate(X, interpolation=InterpolationMode.BILINEAR, angle=theta, expand=expand, fill=fill)
    elif axis == 1:
        X = X.permute((1, 0, 2))
        X = rotate(X, interpolation=InterpolationMode.BILINEAR, angle=theta, expand=expand, fill=fill)
        X = X.permute((1, 0, 2))
    elif axis == 2:
        X = X.permute((2, 1, 0))
        X = rotate(X, interpolation=InterpolationMode.BILINEAR, angle=-theta, expand=expand, fill=fill)
        X = X.permute((2, 1, 0))
    else:
        raise Exception('Not invalid axis')
    return X.squeeze(0)

if __name__ == "__main__":
    input_data = np.ones((300,300,300))
    input_data[0:250,0:250,0:250] = 0.75
    input_data[0:150,0:150,0:150] = 0.5
    input_data[0:50,0:50,0:50] = 0.25
    input_data = np.pad(input_data, ((100, 100), (100, 100), (100, 100)), 'constant', constant_values=((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)))
    theta = 30
    
    output1 = rotation_3d(input_data, 0, theta)
    output2 = rotation_3d(output1, 1, theta)
    output3 = rotation_3d(output1, 2, theta)

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    ax = axes.ravel()
    ax[0].imshow(input_data[120, :, :], cmap='gray')
    ax[0].set_title('Original image')
    ax[1].imshow(output1.cpu()[140, :, :], cmap='gray')
    ax[1].set_title('Rotated image around x axis')
    ax[2].imshow(output2.cpu()[140, :, :], cmap='gray')
    ax[2].set_title('Rotated image around y axis')
    ax[3].imshow(output3.cpu()[140, :, :], cmap='gray')
    ax[3].set_title('Rotated image around z axis')
    fig.suptitle('1')
    plt.show()
