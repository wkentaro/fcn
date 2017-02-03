from fcn.datasets import pascal


def test_pascal():
    dataset = pascal.PascalVOC2012SegmentationDataset('val')
    viz = dataset.visualize_example(0)
    return viz


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    viz = test_pascal()
    plt.imshow(viz)
    plt.show()
