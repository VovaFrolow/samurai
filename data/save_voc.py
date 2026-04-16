import os
import tarfile
import numpy as np
import webdataset as wds
import tqdm
from PIL import Image
import torch.utils.data as data

from utils import mkdir_if_missing, download_file_from_google_drive


class VOC12(data.Dataset):
    GOOGLE_DRIVE_LINK = 'https://drive.google.com/uc?id=1pxhY5vsLwXuz6UHZVUKhtb7EJdCg2kuH'
    FILE = 'PASCAL_VOC.tgz'

    VOC_CATEGORY_NAMES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                          'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                          'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                          'sofa', 'train', 'tvmonitor']

    def __init__(self, root, split='val', download=True):
        self.root = root
        self.folder = 'VOCSegmentation'
        self.split = split
        self.image_dir = os.path.join(self.root, self.folder, 'images')
        
        if split == 'trainaug':
            self.semseg_dir = os.path.join(self.root, self.folder, 'SegmentationClassAug')
        else:
            self.semseg_dir = os.path.join(self.root, self.folder, 'SegmentationClass')

        if download:
            self._download()

        self.images, self.semsegs = self._load_data()

    def _download(self):
        _fpath = os.path.join(self.root, self.FILE)

        if os.path.isfile(os.path.join(self.root, self.folder, "sets")):
            print('Files already downloaded')
            return

        print('Downloading dataset from google drive')
        mkdir_if_missing(os.path.dirname(_fpath))
        download_file_from_google_drive(self.GOOGLE_DRIVE_LINK, _fpath)

        print('Extracting tar file')
        with tarfile.open(_fpath) as tar:
            tar.extractall(path=self.root)
        print('Done!')
        os.remove(_fpath)
        print(f'Successfully deleted the tar file: {_fpath}')

    def _load_data(self):
        if self.split == 'trainval':
            # Комбинирование train и val наборов
            train_images, train_semsegs = self._load_split_data('train')
            val_images, val_semsegs = self._load_split_data('val')
            
            images = train_images + val_images
            semsegs = train_semsegs + val_semsegs
        else:
            images, semsegs = self._load_split_data(self.split)
            
        print(f'Loaded {len(images)} images and {len(semsegs)} segmentation masks for {self.split}')
        return images, semsegs
    
    def _load_split_data(self, split_name):
        """Загружает данные для конкретного сплита"""
        split_file = os.path.join(self.root, self.folder, 'sets', split_name + '.txt')
        images, semsegs = [], []

        with open(split_file, "r") as f:
            lines = f.read().splitlines()

        for line in lines:
            img_path = os.path.join(self.image_dir, line + ".jpg")
            semseg_path = os.path.join(self.semseg_dir, line + '.png')
            assert os.path.isfile(img_path), f"Image file not found: {img_path}"
            assert os.path.isfile(semseg_path), f"Segmentation file not found: {semseg_path}"
            images.append(img_path)
            semsegs.append(semseg_path)

        return images, semsegs

    def get_samples(self, load_annotations=True, convert_images_to_numpy=True):
        for img_path, semseg_path in zip(self.images, self.semsegs):
            key = os.path.basename(img_path).split('.')[0]
            
            if convert_images_to_numpy:
                img_key = "image.npy"
                with Image.open(img_path).convert('RGB') as img:
                    img.load()
                    img_data = np.asarray(img, dtype="uint8")
            else:
                img_key = "image" + os.path.splitext(img_path)[-1]
                with open(img_path, "rb") as f:
                    img_data = f.read()

            sample = {"__key__": key, img_key: img_data}
            
            if load_annotations:
                if convert_images_to_numpy:
                    semseg = np.array(Image.open(semseg_path), dtype=np.uint8)
                    sample["segmentations.npy"] = semseg
                else:
                    with open(semseg_path, "rb") as f:
                        semseg_data = f.read()
                    sample["segmentations.png"] = semseg_data
            
            yield sample


def write_dataset(image_dir, out_dir, split, load_annotations=True, convert_images_to_numpy=True, maxcount=256):
    """
    Write VOC dataset to webdataset shards.
    
    Args:
        image_dir: Directory containing the VOC dataset
        out_dir: Directory where shards will be written
        split: Dataset split to use ('train', 'val', 'trainaug', 'trainval')
        load_annotations: Whether to include segmentation annotations
        convert_images_to_numpy: Whether to convert images to numpy arrays or keep original format
        maxcount: Maximum number of samples per shard
    """
    mkdir_if_missing(out_dir)
    voc = VOC12(root=image_dir, split=split)

    pattern = os.path.join(out_dir, f"voc-{split}-%06d.tar")
    with wds.ShardWriter(pattern, maxcount=maxcount) as sink:
        for sample in tqdm.tqdm(voc.get_samples(load_annotations, convert_images_to_numpy), 
                               total=len(voc.images), 
                               desc=f"Creating {split} shards"):
            sink.write(sample)
    
    print(f"Finished writing shards to {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate sharded dataset from Pascal VOC.")
    parser.add_argument("--download-dir", default="./data", help="Directory to download VOC data")
    parser.add_argument("--out-path", default="./shards", help="Directory where shards are written")
    parser.add_argument("--split", default="val", choices=["train", "val", "trainaug", "trainval"],
                       help="Dataset split to use")
    parser.add_argument("--maxcount", type=int, default=1000,
                       help="Maximum samples per shard")
    parser.add_argument("--original-image-format",
                        action="store_true",
                        help="Whether to keep the orginal image encoding (e.g. jpeg), or convert to numpy")
    parser.add_argument("--load-anno", action="store_true", default=True,
                       help="Load annotation data (segmentation masks)")

    args = parser.parse_args()

    write_dataset(
        image_dir=args.download_dir,
        out_dir=args.out_path,
        split=args.split,
        load_annotations=args.load_anno,
        convert_images_to_numpy=not args.original_image_format,
        maxcount=args.maxcount
    )