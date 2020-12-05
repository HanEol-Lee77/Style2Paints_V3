# dataloader

from torch.utils.data import DataLoader
from sketch2color_dataset import Sketch2ColorDataset


def get_dataloader(
    dataset,
    phase,
    batch_size,
    workers=8,
    input_height=256,
    input_width=256,
    pin_memory, # 재구현 코드에 있어서 추가된 것
    # processed_dir='/home/userB/junsulee/youngin/resources/processed' # 데이터를 채워줘야 하는 부분** #$#$
    ####$#$###############################################################################################################
    processed_dir='/home/ubuntu/data/processed' # 데이터를 채워줘야 하는 부분** #$#$
    ####$#$##############################################################################################################
    ):
    """
    dataset: the name of dataset. ex) 'yumi', 'celeba', 'tag2pix'
    phase: use 'train' for training, 'val' for validation, 'test' for testing
    batch_size: the size of batch
    workers: the number of workers used for making batch
    input_height: the height of input image. Do not touch!
    input_width: the width of input image. Do not touch!
    processed_dir: directory which contains datasets. You do not need to change it as long as working in korea university server.
    """

    assert phase in ['train', 'val', 'test']

    dataset = Sketch2ColorDataset(dataset, phase, input_height, input_width, processed_dir)

    if phase == 'train':
        return DataLoader(
            dataset=dataset,
            num_workers=workers,
            batch_size=batch_size,
            shuffle=True
        )
    elif phase == 'val': 
        return DataLoader(
            dataset=dataset,
            num_workers=workers,
            batch_size=batch_size,
            shuffle=False
        )
    else:
        return DataLoader(
            dataset=dataset,
            num_workers=workers,
            batch_size=batch_size,
            shuffle=False
        )