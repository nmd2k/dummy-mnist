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

### Multiple run from one script
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
Wandb support us to log many type of data such as: Audio, Video, Text Table, HTML, Image, gradient, etc. With a simple log for tracking `training` or `testing` experiment, I use the code like shown in my function `test()`:

```python
def test():
    ...
    test_loss /= len(testloader)
    test_accuracy = 100. * correct / len(testloader.dataset)
    # Log test loss and test accuracy
    wandb.log({'epoch':epoch, 'test loss':test_loss, 'test accuracy': test_accuracy})
```

For further tracking, I recommend check [Wandb doc](https://docs.wandb.ai/guides/track/log) for more detail in logging other media type. For example, logging `Image`:

```python
images = wandb.Image(image_array, caption="Top: Output, Bottom: Input")
# log image
wandb.log({"examples": images}
```

# [Intergrating with Pytorch](#intergrate)
Wandb provides `wandb.watch()` for support PyTorch. It automatically log gradients and the network topology. We can call `watch` and pass in PyTorch model:

```python
import wandb
wandb.init(config=[...])

# wandb watch model
wandb.watch(models=model, criterion=loss_function, log='all', log_freq=10)

    for epoch in epochs:
        model.train()
        for idx, (data, target) in enumerate(trainloader):
            ...
            if idx%log_interval==0:
                wandb.log({"loss": loss})
```

### Options
| Arguments | Options                                                                             |
|-----------|-------------------------------------------------------------------------------------|
| log       | `all`: log histograms of both gradients and parameters                              |
|           | `gradients` : log histograms of gradients (default)                                 |
|           | `parameters` : log histograms of parameters                                         |
|           | `None`                                                                              |
| log_freq  | integer (default 1000): The number of steps between logging gradients/parameters    |                                                                                 
| criterion | loss_function: tracking loss function                                               |                                                                                 

# [Artifact versioning](#version)
Wandb allows us to tracking our Data and Model by the concept of Artifact Versioning. Core Artifact feature:

1. **Upload**: Start tracking and versioning any data (files or directories) with run.log_artifact(). You can also track datasets in a remote filesystem (e.g. cloud storage in S3 or GCP) by reference, using a link or URI instead of the raw contents.
2. **Version**: Define an artifact by giving it a type ("raw_data", "preprocessed_data", "balanced_data") and a name ("imagenet_cats_10K"). When you log the same name again, W&B automatically creates a new version of the artifact with the latest contents.
3. **Alias**: Set an alias like "best" or "production" to highlight the important versions in a lineage of artifacts.
Compare: Select any two versions to browse the contents side-by-side. We're also working on a tool for dataset visualization, learn more here →
4. **Download**: Obtain a local copy of the artifact or verify the contents by reference.

### Data versioning
Wandb can tracking  the contents of individual files and directories, in which we may add, remove, replace, or edit items. Start versioning our data while init a run `wandb.init()`:

Sample code:
```python
run = wandb.init(project="my_project")
my_data = wandb.Artifact("new_dataset", type="raw_data")

# add data in dir
my_data.add_dir("path/to/my/data")

# add data file
# my_data.add_file("path/to/file")
run.log_artifact(my_data)
```

### Model versioning
With automatic saving and versioning, each experiment we run stores the most recently trained model artifact to W&B. We can scroll through all these model versions, annotating and renaming as necessary while maintaining the development history.

In simple example of training a model and log it as an artifact named `demo-model`:
```python
# save trained model as artifact
def train():
    for idx (input, target) in emunerate(train_dataloader):
        ...
        
        trained_weight = wandb.Artifact("demo-model", type="model", description="trained a demo model")

        # save model
        torch.onnx.export(model, input, 'weight.onnx')
        wandb.save('weight.onnx')

    # log artifact
    trained_weight.add_file('weight.onnx')
    run.log_artifact(trained_weight)
```

Load model for inference:
```python
run = wandb.init()
# model name + version 
model_at = run.use_artifact('demo-model'+':latest')
# download 
model_dir = model_at.download()
# load model Pytorch
model = Net()
model.load_state_dict(torch.load(model_dir))

```

**Versioning is automatic**, if an artifact changes, just re-run the same artifact creation script. Wandb will checksum the artifact, identify that something changed, and track the new version. If nothing changes, wandb don't reupload any data or create a new version.

# [Hyperparameter tuning with Sweep](#tuning)
