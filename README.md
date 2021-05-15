# Wandb demo

# [Experiment tracking](#track)
### Install Wandb (W&B):
Install Wandb using `pip`:
```
pip install wandb
```
Or `conda`:
```
conda install -c conda-forge wandb 
```

### Setup wandb
Sign up for a free W&B account in [here](https://wandb.ai/login?signup=true). In the termnial, we can start wandb by using:

```
wandb login
```

`wandb` will ask us for a authorize code, which in [Here](wandb.ai/authorize). Or we also can set an env for auto login by:
```
export WANDB_API_KEY = $OUR_API
```
### Start experiment tracking

To start experiment tracking, we need to intergrate Wandb in our script:
```python
import wandb

run = wandb.init(project="mlops-wandb-demo", tags=["dropout", "cnn"])
```

Where: 
- `wandb.init()`: Initialize a new run with specific name through`project=<name>`.

###Multiple run from one script
If we run multiple time in our script, use `run = wandb.init(reinit=True)` to reinit new run and `run.finish()` to end the run.
```python
import wandb
for x in range(10):
    run = wandb.init(reinit=True)
    for y in range (100):
        wandb.log({"metric": x+y})
    run.finish()
```

### Configure runs
Set the `wandb.config` in our script allows us to save the training hyperparameter, input setting, and other dependency variables for our experiments. `wandb.config` save a dictionary of hyperparameters such as learning rate or model type. The model settings we capture in config are useful later to organize and query the results. 

In the code, we can setup the config with a simple code:
```python
# init wandb
config = dict(
    learning_rate = LEARNING_RATE,
    momentum      = MOMENTUM,
    architecture  = ARCHITECTURE,
    dataset       = DATASET
)

wandb.init(config=config)
```

Or by using `argparse.Namespace`:
```python
wandb.init(config={"lr": 0.1})
wandb.config.epochs = 4

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type=int, default=8, metavar='N',
                     help='input batch size for training (default: 8)')
args = parser.parse_args()
wandb.config.update(args) # adds all of the arguments as config variables
```

### Log Data & Media
Wandb support us to log many type of data such as: Audio, Video, Text Table, HTML, Image, etc. With a simple log for tracking `training` or `testing` experiment, I use the code like shown in my function `test()`:

In `test()`:
```python
def test():
    ...
    test_loss /= len(testloader)
    test_accuracy = 100. * correct / len(testloader.dataset)
    wandb.log({'epoch':epoch, 'test loss':test_loss, 'test accuracy': test_accuracy})
```


# [Artifact versioning](#version)

# [Hyperparameter tuning with Sweep](#tuning)
