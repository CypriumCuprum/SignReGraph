import mmcv
import numpy as np
import random
import warnings
from collections.abc import Sequence
from torch.nn.modules.utils import _pair

from ..builder import PIPELINES


@PIPELINES.register_module()
class Flip:
    """Flip the input images with a probability.

    Reverse the order of elements in the given imgs with a specific direction.
    The shape of the imgs is preserved, but the elements are reordered.

    Required keys are "img_shape", "modality", "imgs" (optional), "keypoint"
    (optional), added or modified keys are "imgs", "keypoint", "flip_direction".
    The Flip augmentation should be placed after any cropping / reshaping
    augmentations, to make sure crop_quadruple is calculated properly.

    Args:
        flip_ratio (float): Probability of implementing flip. Default: 0.5.
        direction (str): Flip imgs horizontally or vertically. Options are
            "horizontal" | "vertical". Default: "horizontal".
        flip_label_map (Dict[int, int] | None): Transform the label of the
            flipped image with the specific label. Default: None.
        left_kp (list[int]): Indexes of left keypoints, used to flip keypoints.
            Default: None.
        right_kp (list[ind]): Indexes of right keypoints, used to flip
            keypoints. Default: None.
    """
    _directions = ['horizontal', 'vertical']

    def __init__(self,
                 flip_ratio=0.5,
                 direction='horizontal',
                 flip_label_map=None,
                 left_kp=None,
                 right_kp=None):
        if direction not in self._directions:
            raise ValueError(f'Direction {direction} is not supported. '
                             f'Currently support ones are {self._directions}')
        self.flip_ratio = flip_ratio
        self.direction = direction
        self.flip_label_map = flip_label_map
        self.left_kp = left_kp
        self.right_kp = right_kp

    def _flip_imgs(self, imgs, modality):
        _ = [mmcv.imflip_(img, self.direction) for img in imgs]
        lt = len(imgs)
        if modality == 'Flow':
            # The 1st frame of each 2 frames is flow-x
            for i in range(0, lt, 2):
                imgs[i] = mmcv.iminvert(imgs[i])
        return imgs

    def _flip_kps(self, kps, kpscores, img_width):
        kp_x = kps[..., 0]
        kp_x[kp_x != 0] = img_width - kp_x[kp_x != 0]
        new_order = list(range(kps.shape[2]))
        if self.left_kp is not None and self.right_kp is not None:
            for left, right in zip(self.left_kp, self.right_kp):
                new_order[left] = right
                new_order[right] = left
        kps = kps[:, :, new_order]
        if kpscores is not None:
            kpscores = kpscores[:, :, new_order]
        return kps, kpscores

    @staticmethod
    def _box_flip(box, img_width):
        """Flip the bounding boxes given the width of the image.

        Args:
            box (np.ndarray): The bounding boxes.
            img_width (int): The img width.
        """
        box_ = box.copy()
        box_[..., 0::4] = img_width - box[..., 2::4]
        box_[..., 2::4] = img_width - box[..., 0::4]
        return box_

    def __call__(self, results):
        """Performs the Flip augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if 'keypoint' in results:
            assert self.direction == 'horizontal', (
                'Only horizontal flips are'
                'supported for human keypoints')

        modality = results['modality']
        if modality == 'Flow':
            assert self.direction == 'horizontal'

        flip = np.random.rand() < self.flip_ratio

        results['flip'] = flip
        results['flip_direction'] = self.direction
        img_width = results['img_shape'][1]

        if self.flip_label_map is not None and flip:
            results['label'] = self.flip_label_map.get(results['label'],
                                                       results['label'])

        if flip:
            if 'imgs' in results:
                results['imgs'] = self._flip_imgs(results['imgs'], modality)
            if 'keypoint' in results:
                kp = results['keypoint']
                kpscore = results.get('keypoint_score', None)
                kp, kpscore = self._flip_kps(kp, kpscore, img_width)
                results['keypoint'] = kp
                if 'keypoint_score' in results:
                    results['keypoint_score'] = kpscore

        if 'gt_bboxes' in results and flip:
            assert self.direction == 'horizontal'
            width = results['img_shape'][1]
            results['gt_bboxes'] = self._box_flip(results['gt_bboxes'], width)
            if 'proposals' in results and results['proposals'] is not None:
                assert results['proposals'].shape[1] == 4
                results['proposals'] = self._box_flip(results['proposals'],
                                                      width)

        return results

    def __repr__(self):
        repr_str = (
            f'{self.__class__.__name__}('
            f'flip_ratio={self.flip_ratio}, direction={self.direction}, '
            f'flip_label_map={self.flip_label_map})')
        return repr_str
    
    
# My augmentation
@PIPELINES.register_module()
class Part_Drop_Sign_Language:
    """Drop the left or right limbs of the skeleton. """

    def __init__(self, p=0.2, left_hand=list, right_hand=list):
        assert isinstance(p, tuple) or isinstance(p, float)
        self.p = p
        self.left_hand = left_hand
        self.right_hand = right_hand
        assert self.p >= 0 and self.p <= 1, f'Drop probability should be between 0 and 1. But got {self.p}.'
        assert self.left_hand is not None, 'Left hand keypoints should be specified.'
        assert self.right_hand is not None, 'Right hand keypoints should be specified.'

    def __call__(self, results):
        skeleton = results['keypoint']
        p = self.p

        if random.random() < p:                   
            part = random.randint(0, 1)    
            temp = skeleton.copy()
            # M T V C -> V M T C
            temp = temp.transpose(2, 0, 1, 3) 
            M, T, V, C = skeleton.shape
            x_new = np.zeros((M, T, C))
            if part == 0:
                for idx in self.left_hand:
                    temp[idx] = x_new
            elif part == 1:
                for idx in self.right_hand:
                    temp[idx] = x_new

            # V M T C -> M T V C
            temp = temp.transpose(1, 2, 0, 3)
            results['keypoint'] = temp
        else:
            results['keypoint'] = skeleton

        return results

@PIPELINES.register_module()
class Horizontal_Flip_Keypoint:
    def __init__(self,
                 flip_ratio=0.5):
        self.flip_ratio = flip_ratio


    def _flip_kps(self, kps, kpscores, img_width=1):
        kp_x = kps[..., 0]
        kp_x[kp_x != 0] = img_width - kp_x[kp_x != 0]
        # new_order = list(range(kps.shape[2]))
        # if self.left_kp is not None and self.right_kp is not None:
        #     for left, right in zip(self.left_kp, self.right_kp):
        #         new_order[left] = right
        #         new_order[right] = left
        # kps = kps[:, :, new_order]
        # if kpscores is not None:
        #     kpscores = kpscores[:, :, new_order]
        # Skip flip keypoint score 2 parts left and right because do not use keypoint score
        return kps, kpscores

    def __call__(self, results):
        """Performs the Flip augmentation.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """

        flip = np.random.rand() < self.flip_ratio

        if flip and 'keypoint' in results:
            kp = results['keypoint']
            kpscore = results.get('keypoint_score', None)
            kp, kpscore = self._flip_kps(kp, kpscore)
            results['keypoint'] = kp
            if 'keypoint_score' in results:
                results['keypoint_score'] = kpscore


        return results
    
@PIPELINES.register_module()
class Frames_Drop:
    def __init__(self, drop_ratio=0.2, max_frames_drop=10):
        self.drop_ratio = drop_ratio
        self.max_frames_drop = max_frames_drop
        
    def __call__(self, results):
        drop_rate = self.drop_ratio
        max_frames_drop = self.max_frames_drop
        
        num_frames_drop = random.randint(1, max_frames_drop)
        skeleton = results['keypoint']
        M, T, V, C = skeleton.shape
        replace_frames = np.zeros((M , V, C))
        if random.random() < drop_rate: 
            temp = skeleton.copy()
            
            # M T V C -> T M V C
            temp = temp.transpose(1, 0, 2, 3)
            frames_drop = random.sample(range(T),num_frames_drop)
            for idx in frames_drop:
                temp[idx] = replace_frames
            # T M V C -> M T V C
            temp = temp.transpose(1, 0, 2, 3)
            results['keypoint'] = temp
        else:
            results['keypoint'] = skeleton

        return results
        
@PIPELINES.register_module()
class Drop_One_hand_Sign_Language:
    """Drop the left or right limbs of the skeleton. """

    def __init__(self, left_hand=list, right_hand=list, is_drop_left=True):
        self.left_hand = left_hand
        self.right_hand = right_hand
        self.is_drop_left = is_drop_left
        assert self.left_hand is not None, 'Left hand keypoints should be specified.'
        assert self.right_hand is not None, 'Right hand keypoints should be specified.'

    def __call__(self, results):
        skeleton = results['keypoint']
                
        temp = skeleton.copy()
        # M T V C -> T V M C
        temp = temp.transpose(1, 2, 0, 3) 
        M, T, V, C = skeleton.shape
        pos_drop = random.randint(T//3, T//2)
        drop_keypoint = skeleton[0, pos_drop]
        x_new = np.zeros((M, C))
        if self.is_drop_left:
            for idx in self.left_hand:
                temp[pos_drop, idx] = x_new
        else:
            for idx in self.right_hand:
                temp[pos_drop, idx] = x_new

        # T V M C -> M T V C
        temp = temp.transpose(2, 0, 1, 3)
        results['keypoint'] = temp
        results['drop_area'] = drop_keypoint.transpose()[:,6:27]

        return results
    
@PIPELINES.register_module()
class Drop_Keypoint_Sign_Language:
    """Drop the left or right limbs of the skeleton. """

    def __init__(self, drop_area=list):
        self.drop_area = drop_area
        assert self.drop_area is not None, 'Droplist keypoints should be specified.'

    def __call__(self, results):
        skeleton = results['keypoint']
                
        temp = skeleton.copy()
        # M T V C -> T V M C
        temp = temp.transpose(1, 2, 0, 3) 
        M, T, V, C = skeleton.shape
        pos_drop = random.randint(T//3, T//2)
        drop_keypoint = skeleton[0, pos_drop]
        x_new = np.zeros((M, C))
        for idx in self.drop_area:
            temp[pos_drop, idx] = x_new
            
        # T V M C -> M T V C
        temp = temp.transpose(2, 0, 1, 3)
        results['keypoint'] = temp
        results['drop_area'] = drop_keypoint.transpose()[:,:]

        return results
    
@PIPELINES.register_module()
class Drop_Keypoint_MultiFrame_Sign_Language:
    """Drop the left or right limbs of the skeleton. """

    def __init__(self, drop_area:list, num_drop_frames:int):
        self.drop_area = drop_area
        self.num_drop_frames = num_drop_frames

    def __call__(self, results):
        skeleton = results['keypoint']
                
        temp = skeleton.copy()
        # M T V C -> T V M C
        temp = temp.transpose(1, 2, 0, 3) 
        M, T, V, C = skeleton.shape
        start_idx_drop = random.randint(T//3, T//2)
        drop_keypoint = skeleton[0, start_idx_drop:start_idx_drop+self.num_drop_frames]
        x_new = np.zeros((M, C))
        for pos_drop in range(start_idx_drop, start_idx_drop+self.num_drop_frames):
            for idx in self.drop_area:
                temp[pos_drop, idx] = x_new
            
        # T V M C -> M T V C
        temp = temp.transpose(2, 0, 1, 3)
        results['keypoint'] = temp
        results['drop_area'] = drop_keypoint.transpose() #  wrong because drop_keypoint is 3D

        return results