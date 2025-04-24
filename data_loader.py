import torch
import torchvision.transforms as transforms
from nltk.tokenize import wordpunct_tokenize
import torch.utils.data as data
from torch.utils import data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab, transform=None, test_mode=False):
        """
        Args:
            root: 图像目录。
            json: COCO 标注文件路径。
            vocab: 词汇表对象。
            transform: 图像预处理。
            test_mode: 如果为 True，则仅加载图像，而不依赖 annotations。
        """
        self.root = root
        self.coco = COCO(json)
        self.vocab = vocab
        self.transform = transform
        self.test_mode = test_mode

        if test_mode:
            self.ids = list(self.coco.imgs.keys())  # 从 `images` 字段加载所有图像 ID
        else:
            self.ids = list(self.coco.anns.keys())  # 从 `annotations` 字段加载所有标注 ID

    def __getitem__(self, index):
        """返回一个数据样本（图像及可选的描述）。"""
        coco = self.coco
        vocab = self.vocab
        img_id = self.ids[index]

        # 加载图像
        path = coco.loadImgs(img_id)[0]['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # 如果是测试模式，只返回图像和图像 ID
        if self.test_mode:
            return image, img_id

        # 如果是训练或验证模式，返回图像和描述
        ann_id = coco.getAnnIds(imgIds=img_id)[0]
        caption = coco.anns[ann_id]['caption']
        tokens = nltk.tokenize.wordpunct_tokenize(str(caption).lower())
        caption = [vocab('<start>')] + [vocab(token) for token in tokens] + [vocab('<end>')]
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.ids)



def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    # print(data)
    # for item in data:
        # print("Type of item:", type(item))
        # print("Content of item:", item)
    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths

def get_loader(root, json, vocab, transform, batch_size, shuffle, num_workers, test_mode=False):
    """返回自定义的 COCO 数据加载器。"""
    coco = CocoDataset(root=root, json=json, vocab=vocab, transform=transform, test_mode=test_mode)

    data_loader = torch.utils.data.DataLoader(dataset=coco,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn if not test_mode else None)
    return data_loader
