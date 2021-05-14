from config import DATA_PATH
import wandb
from utils.utils import get_dataset, get_transform

def download_dataset():
    train_set, test_set = get_dataset(transform=get_transform())

if __name__ == '__main__':
    run = wandb.init(name="mlops-wandb-demo", job_type="dataset-creation")
    artifact = wandb.Artifact('mnist', type='dataset')
    download_dataset()
    artifact.add_dir(DATA_PATH)
    run.log_artifact(artifact)