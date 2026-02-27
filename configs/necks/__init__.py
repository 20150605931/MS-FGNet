from .gap import GlobalAveragePooling
from .part_gap_SP import PartGlobalAveragePooling_SP
from .hr_fuse import HRFuseScales
from .part_gap_recurrent_BP import PartGlobalAveragePooling_recurrent_BP
from .part_gap_recurrent_no_BP import PartGlobalAveragePooling_recurrent_no_BP
from .part_gap_MSP import PartGlobalAveragePooling_MSP
from .part_gap_MSP_BP1 import PartGlobalAveragePooling_MSP_BP1
from .part_gap_recurrent_LBP import PartGlobalAveragePooling_recurrent_LBP

__all__ = ['GlobalAveragePooling', 'HRFuseScales', 'PartGlobalAveragePooling_SP', 'PartGlobalAveragePooling_recurrent_BP',
            'PartGlobalAveragePooling_MSP','PartGlobalAveragePooling_MSP_BP1','PartGlobalAveragePooling_recurrent_no_BP',
            'PartGlobalAveragePooling_recurrent_LBP']
