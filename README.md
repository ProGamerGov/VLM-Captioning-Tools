# VLM-Captioning-Tools
Python scripts to use for captioning images with VLMs

## CogVLM Captioning Tool

This script uses [CogVLM](https://github.com/THUDM/CogVLM) to create captions for one or more directories (including subdirectories) and lists of file paths in txt files.
Repeating sequence failures are automatically detected and resolved by changing model parameters when they arise.

Generated captions are saved in a parquet file with the following structure: `['image_name', 'hash', 'short_caption', 'long_caption', 'resolution']`.

**Usage:**

First adjust the `path_and_save_dir` parameter's input directories and backup directory in the script to your liking. Then run it like this:

```
python cogvlm_captioning_tool.py
```

## Caption Summarization Tool

This script uses the [Dolphin 2.6 Mistral 7b - DPO](https://huggingface.co/cognitivecomputations/dolphin-2.6-mistral-7b-dpo) LLM model to create a shortened version of the `long_caption` value for every item in the provided parquet's `short_caption` column. 

**Usage:**

First adjust the input parquet and output parquet parameters in the script to your liking. Then run it like this:

```
python dolphin_mistral_short_captioning_tool.py
```
