
import numpy as np
from torch.utils.data import Sampler


class meta_batch_sampler(Sampler):
    def __init__(self, data_source, epoch_size, way, support_shot, query_shot, seed):
        self.data_source = data_source
        self.epoch_size = epoch_size
        self.way = way
        self.support_shot = support_shot
        self.query_shot = query_shot
        self.total_shot = self.support_shot + self.query_shot
        self.class2id = self.get_class_id()

        self.seed = seed
        self.rnd = np.random

    def get_class_id(self):
        class2id = {}
        for i, (image_path, class_id) in enumerate(self.data_source.imgs):
            if class_id not in class2id:
                class2id[class_id] = []
            class2id[class_id].append(i)

        return class2id

    def __iter__(self):
        if self.seed is not None:
            self.rnd = np.random.RandomState(self.seed)

        for i in range(self.epoch_size):
            support_index_list = []
            query_index_list = []
            selected_class = self.rnd.choice(list(self.class2id.keys()), size=self.way, replace=False)
            for class_index in selected_class:
                selected_index = self.rnd.choice(self.class2id[class_index], size=self.total_shot, replace=False)
                support_index_list.extend(selected_index[:self.support_shot])
                query_index_list.extend(selected_index[self.support_shot:self.total_shot])
            support_index_list.extend(query_index_list)

            yield support_index_list


