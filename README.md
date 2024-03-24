# VLM-Captioning-Tools
Python scripts to use for captioning images with VLMs

## CogVLM Captioning Tool

This script uses CogVLM to create captions for one or more directories (including subdirectories) and lists of file paths in txt files.
Repeating sequence failures are automatically detected and resolved by changing model parameters when they arise.

Generated captions are saved in a parquet file with the following structure: `['image_name', 'hash', 'short_caption', 'long_caption', 'resolution']`.

Usage:

First adjust the `path_and_save_dir` parameter's input directories and backup directory in the script to your liking. Then run it like this:

```
python cogvlm_captioning_tool.py
```
