import json
from pathlib import Path

from datasets.dataloader import DataLoader


ROOT_DIR = Path(__file__).parent


class WineDataLoader(DataLoader):
    """
    Load the dataset of Wine reviews.
    """
    def __init__(
        self,
        batchsize:int=128,
        seq_len: int=150,
        vocab_size:int=10000,
    ):
        with open(ROOT_DIR / "winemag-data-130k-v2.json", "r") as f:
            data = json.load(f)
        dataset = [x['description'] for x in data]

        super(WineDataLoader, self).__init__(
            dataset=dataset,
            batchsize=batchsize,
            seq_len=seq_len,
            vocab_size=vocab_size,
        )
