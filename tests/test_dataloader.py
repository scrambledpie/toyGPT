from datasets.wine_dataset import WineDataLoader


def test_wine_dataloader():

    for batchsize in [100, 3000]:
        for seq_len in [10, 300]:
            wd = WineDataLoader(batchsize=batchsize, seq_len=seq_len)

            for i, x in enumerate(wd):
                assert x.shape[0] == batchsize
                assert x.shape[1] == seq_len

            assert i == len(wd) - 1
