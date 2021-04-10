# compositional-gscan
This is the code repository for the paper "[Compositional Networks Enable Systematic Generalization for Grounded Language Understanding](https://arxiv.org/abs/2008.02742)"

## Data
After cloning this repo, unzip the `compositional_splits.zip` and `target_length_split.zip` in the `multimodal_seq2seq_gSCAN/data` folder.

## Install
We recommend to run in a new virtual environment. You can clone this repo and run `pip install -r requirements.txt` to get the dependencies.

We use constituency parser from [AllenNLP](https://allennlp.org/) and dependency parser from [Stanza](https://stanfordnlp.github.io/stanza/). You may need to set environment variables `ALLENNLP_CACHE_ROOT` and `STANZA_RESOURCES_DIR` if you store the pretrained parsers to a custom path.

## Run
The training and testing parameters are stored under `configs` folder. 
You can train and test using the following scripts:

```
python train.py --config [path_to_config_file] --k [k] --model_prefix [path_to_model_directory]
```

```
python test.py --config [path_to_config_file] --model_prefix [path_to_model_directory] --output_directory [path_to_output_directory] &> logs.txt
```

You will need to set `--max_seq_length` to 13, 14, or -1 (the max length in the dataset) when training with the `target_length` split.

## Visualization
To visualize the predictions and attention maps, you can use our custom `visualize_prediction` method in `GroundedScan/dataset.py`. It loads the prediciton json file and save the GIFs and attention maps in the folder specified in the argument.

```
from GroundedScan.dataset import GroundedScan
dataset = GroundedScan.load_dataset_from_file(args.data_path + 'dataset.txt', 'tmp_images')
dataset.visualize_prediction('prediction.json', folder='viz')
```
