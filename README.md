# VLM-Captioning-Tools
Python scripts to use for captioning images with VLMs

## CogVLM Captioning Tool

This script uses [CogVLM](https://github.com/THUDM/CogVLM) to create captions for one or more directories (including subdirectories) and lists of file paths in txt files.
Repeating sequence caption failures caused by greedy search algorithms are automatically detected and resolved by changing model parameters when they arise.

Generated captions are saved in a parquet file with the following structure: `['image_name', 'hash', 'short_caption', 'long_caption', 'resolution']`.

**Usage:**

First adjust the `path_and_save_dir` parameter's input directories and backup directory in the script to your liking. Then run it like this:

```
python cogvlm_captioning_tool.py
```

### Prefixes

CogVLM often uses common prefix substrings in output captions. This function provides a way to remove those if desired.

```
def modify_caption(caption: str) -> str:
    """
    Removes common prefix substrings from CogVLM outputs.

    Args:
        caption (str): A string containing a cogvlm caption.

    Returns:
        str: The caption with the prefix substring removed
            or altered if it was present.
    """
    base_words = ['showcases ', 'portrays ', 'appears to be ', 'is ', 'depicts ', 'features ']
    prefix_substrings = [("The image " + s, '') for s in base_words] + [("This image " + s, '') for s in base_words]
    prefix_substrings += [("In this " + s, '') for s in ["picture, ", "depiction, ",  "piece, ", "image, ", "scene, "]]
    prefix_substrings  += [
        ('In this artwork, ', 'Artwork of '),
        ('In this illustration, ', 'Illustration of '),
        ('In this art piece, ', 'Art of ')
    ]
    pattern = '|'.join([re.escape(opening) for opening, _ in prefix_substrings])
    replacers = {opening: replacer for opening, replacer in prefix_substrings}
    
    def replace_fn(match):
        return replacers[match.group(0)]
    
    return re.sub(pattern, replace_fn, caption, count=1, flags=re.IGNORECASE).capitalize()
```

## Caption Summarization Tool

This script uses the [Dolphin 2.6 Mistral 7b - DPO](https://huggingface.co/cognitivecomputations/dolphin-2.6-mistral-7b-dpo) LLM model to create a shortened version of the `long_caption` value for every item in the provided parquet's `short_caption` column. 

**Usage:**

First adjust the input parquet and output parquet parameters in the script to your liking. Then run it like this:

```
python dolphin_mistral_short_captioning_tool.py
```


## Bad Caption Finder Tool

This script provides the ability to determine the number of caption failures in a dataset due to greedy search induced failures. It is capable of detecting repeating characters, words, sentences, and more and thus is extremely robust.

**Usage:**

First ensure the script loads your dataset captions (example code loads from parquet, but this can be changed). Then run it like this:

```
python bad_caption_finder.py
```
