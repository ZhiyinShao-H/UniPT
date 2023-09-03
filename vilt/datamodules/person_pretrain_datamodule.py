from vilt.datasets import PersonPretrainDataset
from .datamodule_base import BaseDataModule


class PersonPretrainDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return PersonPretrainDataset

    @property
    def dataset_cls_no_false(self):
        return PersonPretrainDataset

    @property
    def dataset_name(self):
        return "PP"
