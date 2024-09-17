import datetime
from pathlib import Path


ROOT = Path(__file__).parent.parent

CHECKPOINTS_DIR = ROOT / "checkpoints"
LOGS_DIR = ROOT / "tensorboard_logs"


def make_new_folders() -> tuple[Path, Path]:
    """ Make brand new folders for checkpointing and logging """

    # xp_id: an integer theat increments for eacn new eXPeriment
    # get all the folder xp_id prefixed integers and go one higher
    xp_folders = CHECKPOINTS_DIR.glob("*")
    xp_id = 0
    for folder in xp_folders:
        try:
            xp_id_old = int(folder.stem.split("_")[0])
            xp_id = max(xp_id, xp_id_old + 1)
        except ValueError:
            pass

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    model_name = f"{xp_id}_{timestamp}"

    checkpoint_dir = CHECKPOINTS_DIR / model_name
    checkpoint_dir.mkdir(parents=True)

    log_dir = LOGS_DIR / model_name
    log_dir.mkdir(parents=True)

    print(f"Made new folders: {checkpoint_dir}, {log_dir}")

    return checkpoint_dir, log_dir
