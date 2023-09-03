import json
import os
import pandas as pd
import pyarrow as pa
import random

from tqdm import tqdm
from glob import glob
from collections import defaultdict


def path2rest(path, iid2captions, iid2split):
    name = path.split("/")[-1]
    with open(path, "rb") as fp:
        binary = fp.read()
    captions = iid2captions[name]
    split = iid2split[name]
    return [binary, captions, name, split]


def make_arrow(json_root, dataset_root):
    with open("/root/paddlejob/workspace/env_run/output/shaozhiyin/data/LUPerson/LUPerson/attribute/LuPerson_caption_all_some_2v.json", "r") as fp:
        captions = json.load(fp)

    # captions = captions["images"]

    iid2captions = defaultdict(list)
    iid2split = dict()

    for cap in tqdm(captions):
        filename = cap["file_path"]
        random_num = random.random()
        if random_num < 0.001:
            iid2split[filename] = "test"
        elif 0.001 <= random_num < 0.0015:
            iid2split[filename] = "val"
        elif 0.0015 <= random_num < 0.002:
            iid2split[filename] = "restval"
        else:
            iid2split[filename] = "train"
        for c in cap["captions"]:
            iid2captions[filename].append(c)

    print(len(iid2captions))
    paths = list(glob("/root/paddlejob/workspace/env_run/output/shaozhiyin/data/LUPerson/LUPerson/images/*.jpg"))
    random.shuffle(paths)
    caption_paths = [path for path in paths if path.split("/")[-1] in iid2captions]
    # caption_paths = []

    if len(paths) == len(caption_paths):
        print("all images have caption annotations")
    else:
        print("not all images have caption annotations")
    print(
        len(paths), len(caption_paths), len(iid2captions),
    )

    bs = [path2rest(path, iid2captions, iid2split) for path in tqdm(caption_paths)]

    for split in ["train", "val", "restval", "test"]:
        batches = [b for b in bs if b[-1] == split]

        dataframe = pd.DataFrame(
            batches, columns=["image", "caption", "image_id", "split"],
        )

        table = pa.Table.from_pandas(dataframe)
        os.makedirs(dataset_root, exist_ok=True)
        with pa.OSFile(
            f"{dataset_root}/LUPerson_{split}.arrow", "wb"
        ) as sink:
            with pa.RecordBatchFileWriter(sink, table.schema) as writer:
                writer.write_table(table)
