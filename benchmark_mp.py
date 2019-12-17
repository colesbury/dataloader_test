import argparse
import os
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import tqdm
import random
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


parser = argparse.ArgumentParser(description='ImageNet data loading benchmark')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
class Imagenet(datasets.ImageFolder):
    def __init__(self, dir):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        super().__init__(dir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    # def __len__(self):
    #     # Limit to first 10%
    #     return round(super().__len__() / 10)

def main():
    args = parser.parse_args()

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = Imagenet(traindir)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    for sample, target in tqdm.tqdm(train_loader):
        assert sample.is_pinned()
        del sample
        del target


if __name__ == "__main__":
    torch.manual_seed(1)
    random.seed(1)
    main()