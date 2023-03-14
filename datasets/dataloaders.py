import torch
import torchvision.datasets as datasets
import torch.utils.data
from PIL import Image
from datasets import samplers, transform_manager


def image_loader(path, is_training, transform_type, pre):
    image = Image.open(path)
    image = image.convert('RGB')
    image_transform = transform_manager.get_transform(is_training=is_training, transform_type=transform_type, pre=pre)
    image = image_transform(image)

    return image


def get_dataset(data_path, is_training, transform_type, pre):
    def loader(image_path):
        return image_loader(path=image_path, is_training=is_training, transform_type=transform_type, pre=pre)
    dataset = datasets.ImageFolder(data_path, loader=loader)

    return dataset


def meta_dataloader(data_path, is_training, pre, transform_type, epoch_size, way, support_shot, query_shot, seed):
    dataset = get_dataset(data_path=data_path, is_training=is_training, transform_type=transform_type, pre=pre)
    sampler = samplers.meta_batch_sampler(data_source=dataset, epoch_size=epoch_size, way=way,
                                          support_shot=support_shot, query_shot=query_shot, seed=seed)
    data_loader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=16,  pin_memory=True)

    return data_loader


def normal_dataloader(data_path, transform_type, batch_size):
    dataset = get_dataset(data_path=data_path, is_training=True, transform_type=transform_type, pre=None)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True
    )

    return data_loader
