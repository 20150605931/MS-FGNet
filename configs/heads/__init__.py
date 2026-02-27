from .linear_head import LinearClsHead
from .stacked_head import StackedLinearClsHead
from .cls_head import ClsHead
from .part_cls_head import PartClsHead
from .weighted_part_cls_head_SP import WeightedPartClsHead_SP
from .weighted_part_cls_head_MSP import WeightedPartClsHead_MSP
from .weighted_part_cls_head_MSP_BP1 import WeightedPartClsHead_MSP_BP1
from .weighted_part_cls_head_1 import WeightedPartClsHead_1
from .weighted_part_cls_head_2 import WeightedPartClsHead_2
from .weighted_part_cls_head_3 import WeightedPartClsHead_3
from .weighted_part_cls_head_recurrent import WeightedPartClsHead_recurrent
from .weighted_part_cls_head_recurrent_1 import WeightedPartClsHead_recurrent_1
from .weighted_part_cls_head_recurrent_LBP import WeightedPartClsHead_recurrent_LBP
from .weighted_part_cls_head_recurrent_1_no_BP import WeightedPartClsHead_recurrent_1_no_BP
from .vision_transformer_head import VisionTransformerClsHead
from .deit_head import DeiTClsHead
from .conformer_head import ConformerHead
from .efficientformer_head import EfficientFormerClsHead
from .levit_head import LeViTClsHead


__all__ = [
    'LinearClsHead', 'StackedLinearClsHead','ClsHead',
    'VisionTransformerClsHead', 'DeiTClsHead', 'ConformerHead',
    'EfficientFormerClsHead', 'LeViTClsHead', 'PartClsHead', 'WeightedPartClsHead_SP','WeightedPartClsHead_MSP','WeightedPartClsHead_1',
    'WeightedPartClsHead_2', 'WeightedPartClsHead_3','WeightedPartClsHead_recurrent','WeightedPartClsHead_recurrent_1','WeightedPartClsHead_MSP_BP1',
    'WeightedPartClsHead_recurrent_1_no_BP','WeightedPartClsHead_recurrent_LBP'
]
