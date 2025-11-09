# mmdet/models/roi_heads/__init__.py  (mínimo para Faster R-CNN + FPN)
from .base_roi_head import BaseRoIHead
from .standard_roi_head import StandardRoIHead
from .roi_extractors import SingleRoIExtractor
from .bbox_heads import BBoxHead, Shared2FCBBoxHead

__all__ = [
    'BaseRoIHead',
    'StandardRoIHead',
    'SingleRoIExtractor',
    'BBoxHead', 'Shared2FCBBoxHead',
]
