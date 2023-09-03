from vilt.datasets import LUPersonPretrainDataset
from .datamodule_base import BaseDataModule


class LUPersonPretrainDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return LUPersonPretrainDataset

    @property
    def dataset_cls_no_false(self):
        return LUPersonPretrainDataset

    @property
    def dataset_name(self):
        return "LPP"
