import torchvision
import torch
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode
import random 
import numbers
import numpy as np
from PIL import Image
import collections

#
#  Extended Transforms for Semantic Segmentation
#
class ExtRandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(lbl)
        return img, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)



class ExtCompose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, lbl):
        for t in self.transforms:
            img, lbl = t(img, lbl)
        return img, lbl

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ExtCenterCrop(object):
    """Crops the given PIL Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be cropped.
        Returns:
            PIL Image: Cropped image.
        """
        return F.center_crop(img, self.size), F.center_crop(lbl, self.size)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


class ExtRandomScale(object):
    def __init__(self, scale_range, interpolation=InterpolationMode.BILINEAR):
        self.scale_range = scale_range
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be scaled.
            lbl (PIL Image): Label to be scaled.
        Returns:
            PIL Image: Rescaled image.
            PIL Image: Rescaled label.
        """
        assert img.size == lbl.size
        scale = random.uniform(self.scale_range[0], self.scale_range[1])
        target_size = ( int(img.size[1]*scale), int(img.size[0]*scale) )
        return F.resize(img, target_size, self.interpolation), F.resize(lbl, target_size, InterpolationMode.NEAREST)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)

class ExtScale(object):
    """Resize the input PIL Image to the given scale.
    Args:
        Scale (sequence or int): scale factors
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.InterpolationMode.BILINEAR``
    """

    def __init__(self, scale, interpolation=InterpolationMode.BILINEAR):
        self.scale = scale
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be scaled.
            lbl (PIL Image): Label to be scaled.
        Returns:
            PIL Image: Rescaled image.
            PIL Image: Rescaled label.
        """
        assert img.size == lbl.size
        target_size = ( int(img.size[1]*self.scale), int(img.size[0]*self.scale) ) # (H, W)
        return F.resize(img, target_size, self.interpolation), F.resize(lbl, target_size, InterpolationMode.NEAREST)

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str)


class ExtRandomRotation(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({InterpolationMode.NEAREST, InterpolationMode.BILINEAR, InterpolationMode.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.InterpolationMode.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, resample=False, expand=False, center=None, prob=1):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees
        self.prob =  prob

        self.resample = resample
        self.expand = expand
        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, lbl):
        """
            img (PIL Image): Image to be rotated.
            lbl (PIL Image): Label to be rotated.
        Returns:
            PIL Image: Rotated image.
            PIL Image: Rotated label.
        """
        if random.random() <= self.prob:
            angle = self.get_params(self.degrees)
            return F.rotate(img, angle, self.resample, self.expand, self.center), F.rotate(lbl, angle, InterpolationMode.NEAREST, self.expand, self.center)
        else:
            return img, lbl

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        format_string += ', resample={0}'.format(self.resample)
        format_string += ', expand={0}'.format(self.expand)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string



class ExtDiscreteRandomRotation(object):
    """Rotate the image by fixed angles."""

    def __init__(self, degrees=[0,90,180,270]):
        if len(degrees) < 2:
            raise ValueError("Len of list degrees must be at least 2.")
        self.degrees = degrees

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.choice(degrees)
        return angle

    def __call__(self, img, lbl):
        """
            img (PIL Image): Image to be rotated.
            lbl (PIL Image): Label to be rotated.
        Returns:
            PIL Image: Rotated image.
            PIL Image: Rotated label.
        """
        angle = self.get_params(self.degrees)
        return F.rotate(img, angle), F.rotate(lbl, angle)

    def __repr__(self):
        return 'ToDo'
        # format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        # format_string += ', resample={0}'.format(self.resample)
        # format_string += ', expand={0}'.format(self.expand)
        # if self.center is not None:
        #     format_string += ', center={0}'.format(self.center)
        # format_string += ')'
        # return format_string

class ExtRandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
        """
        if random.random() < self.p:
            return F.hflip(img), F.hflip(lbl)
        return img, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)


class ExtRandomVerticalFlip(object):
    """Vertically flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be flipped.
            lbl (PIL Image): Label to be flipped.
        Returns:
            PIL Image: Randomly flipped image.
            PIL Image: Randomly flipped label.
        """
        if random.random() < self.p:
            return F.vflip(img), F.vflip(lbl)
        return img, lbl

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class ExtPad(object):
    def __init__(self, diviser=32):
        self.diviser = diviser
    
    def __call__(self, img, lbl):
        h, w = img.size
        ph = (h//32+1)*32 - h if h%32!=0 else 0
        pw = (w//32+1)*32 - w if w%32!=0 else 0
        im = F.pad(img, ( pw//2, pw-pw//2, ph//2, ph-ph//2) )
        lbl = F.pad(lbl, ( pw//2, pw-pw//2, ph//2, ph-ph//2))
        return im, lbl

class ExtToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, normalize=True, target_type='uint8'):
        self.normalize = normalize
        self.target_type = target_type
    def __call__(self, pic, lbl):
        """
        Note that labels will not be normalized to [0, 1].
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
            lbl (PIL Image or numpy.ndarray): Label to be converted to tensor. 
        Returns:
            Tensor: Converted image and label
        """
        
        lbl = torch.from_numpy( np.array( lbl, dtype=self.target_type) ) if lbl is not None else lbl
        if self.normalize:
            return F.to_tensor(pic), lbl
        else:
            return torch.from_numpy( np.array( pic, dtype=np.float32).transpose(2, 0, 1) ), lbl

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ExtNormalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, lbl):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
            tensor (Tensor): Tensor of label. A dummy input for ExtCompose
        Returns:
            Tensor: Normalized Tensor image.
            Tensor: Unchanged Tensor label
        """
        return F.normalize(tensor, self.mean, self.std), lbl

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ExtRandomCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed

    @staticmethod
    def get_params(img, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be cropped.
            lbl (PIL Image): Label to be cropped.
        Returns:
            PIL Image: Cropped image.
            PIL Image: Cropped label.
        """
        assert img.size == lbl.size, 'size of img and lbl should be the same. %s, %s'%(img.size, lbl.size)
        if self.padding > 0:
            img = F.pad(img, self.padding)
            lbl = F.pad(lbl, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, padding=int((1 + self.size[1] - img.size[0]) / 2))
            lbl = F.pad(lbl, padding=int((1 + self.size[1] - lbl.size[0]) / 2))

        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2))
            lbl = F.pad(lbl, padding=int((1 + self.size[0] - lbl.size[1]) / 2))

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(lbl, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

class ExtRandomScaledCrop(object):
    """Crop the given PIL Image at a random location.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, scale_min=0.5, scale_max=2, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.scale_min = scale_min
        self.scale_max = scale_max

    @staticmethod
    def get_params(img, output_size, scale_min, scale_max):
        """Get parameters for ``crop`` for a random crop.
        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        # print('img size', img.size)
        # print('output size', output_size)
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        scale = np.random.uniform(scale_min, scale_max)
        i = random.randint(0, h - int(th*scale))
        j = random.randint(0, w - int(tw*scale))
        return i, j, th, tw, scale

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be cropped.
            lbl (PIL Image): Label to be cropped.
        Returns:
            PIL Image: Cropped image.
            PIL Image: Cropped label.
        """
        assert img.size == lbl.size, 'size of img and lbl should be the same. %s, %s'%(img.size, lbl.size)
        if self.padding > 0:
            img = F.pad(img, self.padding)
            lbl = F.pad(lbl, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, padding=int((1 + self.size[1] - img.size[0]) / 2))
            lbl = F.pad(lbl, padding=int((1 + self.size[1] - lbl.size[0]) / 2))

        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2))
            lbl = F.pad(lbl, padding=int((1 + self.size[0] - lbl.size[1]) / 2))

        i, j, h, w, scale = self.get_params(img, self.size, self.scale_min, self.scale_max)
        hs, ws = int(h*scale), int(w*scale)
        return F.resized_crop(img, i, j, hs, ws, (h,w)),  F.resized_crop(lbl, i, j, hs, ws, (h,w), interpolation=InterpolationMode.NEAREST)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)

# class ExtRandomScaledCropPad(object):
#     """Crop the given PIL Image at a random location.
#     Args:
#         size (sequence or int): Desired output size of the crop. If size is an
#             int instead of sequence like (h, w), a square crop (size, size) is
#             made.
#         padding (int or sequence, optional): Optional padding on each border
#             of the image. Default is 0, i.e no padding. If a sequence of length
#             4 is provided, it is used to pad left, top, right, bottom borders
#             respectively.
#         pad_if_needed (boolean): It will pad the image if smaller than the
#             desired size to avoid raising an exception.
#     """

#     def __init__(self, size, scale_min=0.5, scale_max=2, padding=0, pad_if_needed=False):
#         if isinstance(size, numbers.Number):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size
#         self.padding = padding
#         self.pad_if_needed = pad_if_needed
#         self.scale_min = scale_min
#         self.scale_max = scale_max

#     @staticmethod
#     def get_params(img, output_size, scale_min, scale_max):
#         """Get parameters for ``crop`` for a random crop.
#         Args:
#             img (PIL Image): Image to be cropped.
#             output_size (tuple): Expected output size of the crop.
#         Returns:
#             tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
#         """
#         # print('img size', img.size)
#         # print('output size', output_size)
#         w, h = img.size
#         th, tw = output_size
#         if w == tw and h == th:
#             return 0, 0, h, w

#         scale = np.random.uniform(scale_min, scale_max)
#         i = random.randint( min(0, h-int(th*scale)), h - int(th*scale))
#         j = random.randint( min(0,w-int(tw*scale)), w - int(tw*scale))
#         return i, j, th, tw, scale

#     def __call__(self, img, lbl):
#         """
#         Args:
#             img (PIL Image): Image to be cropped.
#             lbl (PIL Image): Label to be cropped.
#         Returns:
#             PIL Image: Cropped image.
#             PIL Image: Cropped label.
#         """
#         assert img.size == lbl.size, 'size of img and lbl should be the same. %s, %s'%(img.size, lbl.size)
#         if self.padding > 0:
#             img = F.pad(img, self.padding)
#             lbl = F.pad(lbl, self.padding)

#         # pad the width if needed
#         if self.pad_if_needed and img.size[0] < self.size[1]:
#             img = F.pad(img, padding=int((1 + self.size[1] - img.size[0]) / 2))
#             lbl = F.pad(lbl, padding=int((1 + self.size[1] - lbl.size[0]) / 2))

#         # pad the height if needed
#         if self.pad_if_needed and img.size[1] < self.size[0]:
#             img = F.pad(img, padding=int((1 + self.size[0] - img.size[1]) / 2))
#             lbl = F.pad(lbl, padding=int((1 + self.size[0] - lbl.size[1]) / 2))

#         imw, imh = img.size
#         i, j, h, w, scale = self.get_params(img, self.size, self.scale_min, self.scale_max)
#         hs, ws = int(h*scale), int(w*scale)

#         # adjust for crop scale
        

#         #pad if needed

#         # if i < 0 or j < 0:
#         v = abs(min([i, j, -(i + hs - imh +1 ), - (j + ws - imw +1),  0]))
#         # print('v', v)
#         img = F.pad(img, padding=int(v))
#         lbl = F.pad(lbl, padding=int(v))
#         return F.resized_crop(img, i, j, hs, ws, (h,w)),  F.resized_crop(lbl, i, j, hs, ws, (h,w))

#     def __repr__(self):
#         return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)
from PIL import Image as pimg
RESAMPLE = pimg.BICUBIC
RESAMPLE_D = pimg.BILINEAR

def crop_and_scale_img(img: pimg, crop_box, target_size, pad_size, resample, blank_value):
    target = pimg.new(img.mode, pad_size, color=blank_value)
    target.paste(img)
    res = target.crop(crop_box).resize(target_size, resample=resample)
    return res

class ExtRandomSquareCropAndScale:
    def __init__(self, wh, ignore_id, mean,min=.5, max=2., scale_method=lambda scale, wh, size: int(scale * wh)):
        self.wh = wh
        self.min = min
        self.max = max
        self.ignore_id = ignore_id
        self.random_gens = [self._rand_location]
        self.scale_method = scale_method
        self.mean = mean

    def _random_instance(self, name, W, H):
        def weighted_random_choice(choices):
            max = sum(choices)
            pick = random.uniform(0, max)
            key, current = 0, 0.
            for key, value in enumerate(choices):
                current += value
                if current > pick:
                    return key
                key += 1
            return key

        instances = self.class_instances[name]
        possible_classes = list(set(self.p_class.keys()).intersection(instances.keys()))
        roulette = []
        flat_instances = []
        for c in possible_classes:
            flat_instances += instances[c]
            roulette += [self.p_class[c]] * len(instances[c])
        if len(flat_instances) == 0:
            return [0, W - 1, 0, H - 1]
        index = weighted_random_choice(roulette)
        return flat_instances[index]

    def _rand_location(self, W, H, target_wh, *args, **kwargs):
        try:
            w = np.random.randint(0, W - target_wh + 1)
            h = np.random.randint(0, H - target_wh + 1)
        except ValueError:
            print(f'Exception in RandomSquareCropAndScale: {target_wh}')
            w = h = 0
        # left, upper, right, lower)
        return w, h, w + target_wh, h + target_wh

    def _trans(self, img: pimg, crop_box, target_size, pad_size, resample, blank_value):
        return crop_and_scale_img(img, crop_box, target_size, pad_size, resample, blank_value)

    def __call__(self, img, lbl):
        scale = np.random.uniform(self.min, self.max)
        W, H = img.size
        box_size = self.scale_method(scale, self.wh, img.size)
        pad_size = (max(box_size, W), max(box_size, H))
        target_size = (self.wh, self.wh)
        crop_fn = random.choice(self.random_gens)
        crop_box = crop_fn(pad_size[0], pad_size[1], box_size)

        # PIL values are 0-255
        img = self._trans(img, crop_box, target_size, pad_size, RESAMPLE, (int(self.mean[0]*255),int(self.mean[1]*255),int(self.mean[2]*255)))
        lbl = self._trans(lbl, crop_box, target_size, pad_size, pimg.NEAREST, self.ignore_id)
        return img,lbl




class ExtResize(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.InterpolationMode.BILINEAR``
    """

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        lbl = F.resize(lbl, self.size, InterpolationMode.NEAREST) if lbl is not None else lbl
        return F.resize(img, self.size, self.interpolation), lbl

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str) 

class ExtResizeIm(object):
    """Resize the input PIL Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.InterpolationMode.BILINEAR``
    """

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Image to be scaled.
        Returns:
            PIL Image: Rescaled image.
        """
        return F.resize(img, self.size, self.interpolation), lbl

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, interpolate_str) 

class ExtColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - value, center + value]
            if clip_first_on_zero:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))

        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        # or (0., 0.) for hue, do nothing
        if value[0] == value[1] == center:
            value = None
        return value

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img, lbl):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        return transform(img), lbl

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert callable(lambd), repr(type(lambd).__name__) + " object is not callable"
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string