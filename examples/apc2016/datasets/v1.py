from base import APC2016DatasetBase
from jsk import APC2016jskDataset
from rbo import APC2016rboDataset


class APC2016DatasetV1(APC2016DatasetBase):

    def __init__(self, data_type):
        assert data_type in ('train', 'val')
        self.datasets = [
            APC2016jskDataset(data_type),
            APC2016rboDataset(data_type),
        ]

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def get_example(self, i):
        skipped = 0
        for dataset in self.datasets:
            current_index = i - skipped
            if current_index < len(dataset):
                return dataset.get_example(current_index)
            skipped += len(dataset)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset_train = APC2016DatasetV1('train')
    dataset_val = APC2016DatasetV1('val')
    print('train: %d, val: %d' % (len(dataset_train), len(dataset_val)))
    for i in xrange(len(dataset_val)):
        viz = dataset_val.visualize_example(i)
        plt.imshow(viz)
        plt.show()
