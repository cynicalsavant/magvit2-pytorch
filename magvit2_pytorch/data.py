from pathlib import Path
from functools import partial

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as PytorchDataLoader

import cv2
from PIL import Image
from torchvision import transforms as T, utils

from beartype import beartype
from beartype.typing import Tuple, List
from beartype.door import is_bearable

import numpy as np

from einops import rearrange

import xarray as xr
import pickle

# helper functions

def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def pair(val):
    return val if isinstance(val, tuple) else (val, val)

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def cast_num_frames(t, *, frames):
    f = t.shape[-3]

    if f == frames:
        return t

    if f > frames:
        return t[..., :frames, :, :]

    return pad_at_dim(t, (0, frames - f), dim = -3)

def convert_image_to_fn(img_type, image):
    if not exists(img_type) or image.mode == img_type:
        return image

    return image.convert(img_type)

def append_if_no_suffix(path: str, suffix: str):
    path = Path(path)

    if path.suffix == '':
        path = path.parent / (path.name + suffix)

    assert path.suffix == suffix, f'{str(path)} needs to have suffix {suffix}'

    return str(path)

# channel to image mode mapping

CHANNEL_TO_MODE = {
    1: 'L',
    3: 'RGB',
    4: 'RGBA'
}

# image related helpers fnuctions and dataset

class VSRDataset(Dataset):
    
    def __init__(self, mode, length, logscale = False):
        '''
        Args:
            channels (list): list of channels to use
            mode (str): train or val
            length (int): length of sequence
            logscale (bool): whether to logscale the data
            multi (bool): whether to use multi-channel data
        '''

        ENSEMBLE = 11

        # load data
        self.y = {}

        PATH = "/extra/ucibdl0/shared/data/fv3gfs"

        for member in range(1, ENSEMBLE + 1):

            self.y[member] = xr.open_zarr(f"{PATH}/c384_precip_ave/{member:04d}/sfc_8xdaily_ave.zarr")
        
        # expected sequence length
        self.length = length

        self.mode = mode
        self.logscale = logscale

        self.time_steps = self.y[1].time.shape[0]
        self.tiles = self.y[1].tile.shape[0]

        # load statistics
        with open("/home/prakhs2/fv3net/projects/super_res/data/ensemble_c384_trainstats/chl.pkl", 'rb') as f:

            self.c384_chl = pickle.load(f)

        with open("/home/prakhs2/fv3net/projects/super_res/data/ensemble_c384_trainstats/log_chl.pkl", 'rb') as f:

            self.c384_log_chl = pickle.load(f)

        self.c384_channels = ["PRATEsfc"]

        self.indices = list(range(self.time_steps - self.length + 1))

    def __len__(self):
        
        return len(self.indices)
    
    def __getitem__(self, idx):
        
        time_idx = self.indices[idx]

        if self.mode == 'train':
            
            np.random.seed()
            tile = np.random.randint(self.tiles)
            member = np.random.randint(10) + 1
        
        else:
            
            tile = idx % self.tiles
            member = 11

        y = self.y[member].isel(time = slice(time_idx, time_idx + self.length), tile = tile)

        y = np.stack([y[channel].values for channel in self.c384_channels], axis = 1)
        
        if self.logscale:

            y = np.log(y - self.c384_chl["PRATEsfc"]['min'] + 1e-14)
            y = (y - self.c384_log_chl["PRATEsfc"]['min']) / (self.c384_log_chl["PRATEsfc"]['max'] - self.c384_log_chl["PRATEsfc"]['min'])

        else:

            y = (y - self.c384_chl["PRATEsfc"]['min']) / (self.c384_chl["PRATEsfc"]['max'] - self.c384_chl["PRATEsfc"]['min'])

        return y

# tensor of shape (channels, frames, height, width) -> gif

# handle reading and writing gif

def seek_all_images(img: Tensor, channels = 3):
    mode = CHANNEL_TO_MODE.get(channels)

    assert exists(mode), f'channels {channels} invalid'

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

# tensor of shape (channels, frames, height, width) -> gif

@beartype
def video_tensor_to_gif(
    tensor: Tensor,
    path: str,
    duration = 120,
    loop = 0,
    optimize = True
):
    path = append_if_no_suffix(path, '.gif')
    images = map(T.ToPILImage(), tensor.unbind(dim = 1))
    first_img, *rest_imgs = images
    first_img.save(str(path), save_all = True, append_images = rest_imgs, duration = duration, loop = loop, optimize = optimize)
    return images

# gif -> (channels, frame, height, width) tensor

def gif_to_tensor(
    path: str,
    channels = 3,
    transform = T.ToTensor()
):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, channels = channels)))
    return torch.stack(tensors, dim = 1)

# handle reading and writing mp4

def video_to_tensor(
    path: str,              # Path of the video to be imported
    num_frames = -1,        # Number of frames to be stored in the output tensor
    crop_size = None
) -> Tensor:                # shape (1, channels, frames, height, width)

    video = cv2.VideoCapture(path)

    frames = []
    check = True

    while check:
        check, frame = video.read()

        if not check:
            continue

        if exists(crop_size):
            frame = crop_center(frame, *pair(crop_size))

        frames.append(rearrange(frame, '... -> 1 ...'))

    frames = np.array(np.concatenate(frames[:-1], axis = 0))  # convert list of frames to numpy array
    frames = rearrange(frames, 'f h w c -> c f h w')

    frames_torch = torch.tensor(frames).float()

    frames_torch /= 255.
    frames_torch = frames_torch.flip(dims = (0,)) # BGR -> RGB format

    return frames_torch[:, :num_frames, :, :]

@beartype
def tensor_to_video(
    tensor: Tensor,        # Pytorch video tensor
    path: str,             # Path of the video to be saved
    fps = 25,              # Frames per second for the saved video
    video_format = 'MP4V'
):
    path = append_if_no_suffix(path, '.mp4')

    tensor = tensor.cpu()

    num_frames, height, width = tensor.shape[-3:]

    fourcc = cv2.VideoWriter_fourcc(*video_format) # Changes in this line can allow for different video formats.
    video = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    frames = []

    for idx in range(num_frames):
        numpy_frame = tensor[:, idx, :, :].numpy()
        numpy_frame = np.uint8(rearrange(numpy_frame, 'c h w -> h w c'))
        video.write(numpy_frame)

    video.release()

    cv2.destroyAllWindows()

    return video

def crop_center(
    img: Tensor,
    cropx: int,      # Length of the final image in the x direction.
    cropy: int       # Length of the final image in the y direction.
) -> Tensor:
    y, x, c = img.shape
    startx = x // 2 - cropx // 2
    starty = y // 2 - cropy // 2
    return img[starty:(starty + cropy), startx:(startx + cropx), :]

# override dataloader to be able to collate strings

def collate_tensors_and_strings(data):
    if is_bearable(data, List[Tensor]):
        return (torch.stack(data),)

    data = zip(*data)
    output = []

    for datum in data:
        if is_bearable(datum, Tuple[Tensor, ...]):
            datum = torch.stack(datum)
        elif is_bearable(datum, Tuple[str, ...]):
            datum = list(datum)
        else:
            raise ValueError('detected invalid type being passed from dataset')

        output.append(datum)

    return tuple(output)