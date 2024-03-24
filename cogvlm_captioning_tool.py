import torch
import requests
import pandas as pd
from PIL import Image, ImageFile
from transformers import AutoModelForCausalLM, LlamaTokenizer
from hashlib import md5
import pyarrow.parquet as pq
import os
from pathlib import Path
from itertools import chain
from tqdm import tqdm
import re
from collections import OrderedDict

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 1000000000 # Support gigapixel images


import numpy as np
class LRSCalculationWorker:
    """
    Failed caption detection concept by Zippy (github.com/aredden) & ProGamerGov with help from uptightmoose
    This class utilizes dynamic programming to compute the Longest Repeating Subsequence (LRS)
    of a given string efficiently. Repeating sequences in outputs are suspected to be caused by
    greedy search algorithms.

    Cropping the final 128 characters from a string works the best according to testing, with captions scoring
    above the threshold of 70.0 being likely failures
    """

    def __init__(self) -> None:
        from numba import jit

        @jit(nopython=True, cache=True) # Remove if not using numba JIT
        def lrs(s1, i, j, dp):
            """
            Compute the Longest Repeating Subsequence (LRS) of a given string.

            Args:
                s1 (str): The input string.
                i (int): Starting index for comparison.
                j (int): Starting index for comparison.
                dp (2D array): Dynamic programming table to store computed results.
            Returns:
                int: The length of the Longest Repeating Subsequence.
            Notes:
                This function uses memoization (dynamic programming) to efficiently compute
                the Longest Repeating Subsequence (LRS) of a given string.
            """

            # return if we have reached the
            # end of either string
            if i >= len(s1) or j >= len(s1):
                return 0

            if dp[i][j] != -1:
                return dp[i][j]

            # while dp[i][j] is not computed earlier
            if dp[i][j] == -1:

                # if characters at index m and n matches
                # and index is different
                # Index should not match
                if s1[i] == s1[j] and i != j:
                    dp[i][j] = 1 + lrs(s1, i + 1, j + 1, dp)

                # else if characters at index m and n don't match
                else:
                    dp[i][j] = max(lrs(s1, i, j + 1, dp), lrs(s1, i + 1, j, dp))

            # return answer
            return dp[i][j]

        self.lrs_fn = lrs

    def find_lrs(self, string):
        """
        Find the Longest Repeating Subsequence (LRS) of a given string.

        Args:
            string (str): The input string.
        Returns:
            tuple: A tuple containing the input string and the length of its Longest Repeating Subsequence.
        Notes:
            This method utilizes the previously initialized lrs_fn to compute the LRS.
        """
        leng = len(string) + 1
        arr = np.zeros((leng, leng))
        arr.fill(-1)
        return string, self.lrs_fn(string[-128:], 0, 0, arr)


def load_images_from_directory(directory_path, allowed_extensions=['.jpg', '.jpeg', '.png']):
    """
    Load images from a directory with specified allowed extensions.

    Args:
        directory_path (str): The path to the directory containing the images.
        allowed_extensions (list[str], optional): List of allowed extensions. Defaults to ['.jpg', '.jpeg', '.png'].

    Returns:
        List[str]: A list of paths to the images.
    """
    all_files_with_exts = chain(*[Path(directory_path).rglob("*"+ext) for ext in allowed_extensions])
    filtered_list = list(map(str, all_files_with_exts))
    return list(OrderedDict.fromkeys(filtered_list))


def load_file(list_file, allowed_extensions=['.jpg', '.jpeg', '.png']):
    """
    Load filenames from a text file with specified allowed extensions.

    Args:
        list_file (str): The path to the text file containing filenames.
        allowed_extensions (list[str], optional): List of allowed extensions. Defaults to ['.jpg', '.jpeg', '.png'].

    Returns:
        list[str]: A list of filenames with allowed extensions.
    """
    with open(list_file, 'r') as file:
        filenames = file.read().splitlines()
    filenames = [f.strip() for f in filenames if f != ""]
    filenames = [f for f in filenames if f.endswith(tuple(allowed_extensions))]
    return filenames


def generate_random_gen_kwargs():
    """Randomize input args to avoid failure"""
    import random
    return {
        "max_length": 2048,
        "do_sample": True,
        "top_k": 10,
        "temperature": random.random() * 0.7,
    }


def load_images_and_parquet(path_and_save_dir, parquet_name="output.parquet", allowed_extensions=['.jpg', '.jpeg', '.png'], verbose=True):
    """
    Load images from specified paths and create or append to a parquet file.

    Args:
        path_and_save_dir (list[str] or list[list[str], str]): A tuple containing the path to the image files or directories and the
            directory where the parquet file will be saved.
        parquet_name (str, optional): Name of the parquet file. Default is 'output.parquet'.
        allowed_extensions (list[str], optional): List of allowed image file extensions. Default is ['.jpg', '.jpeg', '.png'].
        verbose (bool, optional): Whether to print verbose messages. Default is True.

    Returns:
        list[str]: A list of loaded image paths.

    """
    path_s = path_and_save_dir[0]
    backup_dir = path_and_save_dir[1]

    if verbose:
        print("Processing", path_s)

    output_file = os.path.join(backup_dir, parquet_name)
    if os.path.exists(output_file):
        df = pd.read_parquet(output_file)
        parquet_exists = True
    else:
        df = pd.DataFrame(columns=['image_name', 'hash', 'short_caption', 'long_caption', 'resolution'])
        parquet_exists = False

    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    path_s = [path_s] if type(path_s) == str else path_s
    images_list = []
    for s in path_s:
        if type(s) == str and s.endswith(".txt"):
            images_list += load_file(s, allowed_extensions=allowed_extensions)
        else:
            images_list += load_images_from_directory(s, allowed_extensions=allowed_extensions)

    if parquet_exists:
        if verbose:
            print("Found existing parquet")
            print("Found", len(images_list), "images before duplicate removal")
        if verbose:
            print("Found", len(images_list), "images after duplicate removal")
        if verbose:
            column_length = len(df['image_name'])
            print("Already processed", column_length, "images")
    
    return images_list


def run_model(model, tokenizer, df, images_list, backup_dir, query_prompt, backup_iter=20, failure_threshold=70.0):
    """
    Run a model for generating captions for a dataset of images.

    Args:
        model (nn.Module): The caption generation model.
        tokenizer (Tokenizer): Tokenizer for the model.
        df (pandas.DataFrame): DataFrame to update with image captions.
        images_list (list[str]): List of paths to images.
        backup_dir (str): Directory to store backup files.
        query_prompt (str): Prompt for generating captions.
        backup_iter (int, optional): Interval for creating backups. Defaults to 20.
        failure_threshold (float, optional): Threshold for caption repeat detection. Defaults to 70.0.
    """
    device = torch.device("cuda")
    column_length = len(df['image_name'])
    skip_total = 0
    lrs = LRSCalculationWorker()

    for index, image_path in tqdm(enumerate(images_list), desc="Processing images", total=len(images_list)):
        if not image_path:
            continue
    
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Failed to load image {image_path}: {e}")
            continue

        image_hash = md5(image.tobytes()).hexdigest()
        resolution = f"{image.size[0]}x{image.size[1]}"  
    
        if df['hash'].str.contains(re.escape(os.path.basename(image_hash))).any():
            print("Image hash already exists in saved captions, skipping...")
            skip_total +=1
            continue

        caption = ''

        inputs = model.build_conversation_input_ids(tokenizer, query=query_prompt, history=[], images=[image])  
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to(device),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to(device),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to(device),
            'images': [[inputs['images'][0].to(device).to(torch.bfloat16)]],
        }
        gen_kwargs = {"max_length": 2048, "do_sample": False}
        gen_kwargs = generate_random_gen_kwargs()

        max_attempts = 5 if failure_threshold else 0
        count = 0
        for _ in range(max_attempts):
            count += 1
            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, inputs['input_ids'].shape[1]:]
                caption = tokenizer.decode(outputs[0])
                caption = caption.replace("</s>", "").strip()
            if not failure_threshold or failure_threshold and lrs.find_lrs(caption[-128:])[1] < failure_threshold or count >= max_attempts:
                break
            else:
                print(f'Got bad caption "{caption}", retrying...')
                gen_kwargs = generate_random_gen_kwargs()
               

        new_row = pd.DataFrame([[image_path, image_hash, '', caption, resolution]], columns=['image_name', 'hash', 'short_caption', 'long_caption', 'resolution'])
        df = pd.concat([df, new_row], ignore_index=True)
    
        index = (index - skip_total) + column_length
        if index % backup_iter == 0 and index > 0:
            backup_file = f'output_backup_{index}.parquet'
            df.to_parquet(os.path.join(backup_dir, backup_file))
    
        df.to_parquet(output_file)



# Initialize the tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
model = AutoModelForCausalLM.from_pretrained(
    'THUDM/cogvlm-chat-hf',
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    load_in_4bit=True,
    trust_remote_code=True
).eval()

# Path to images and outputs dir in the format of: ['path/to/images', 'output_dir'] or 
# [['path/to/images', 'path/to/images'], 'output_dir'] or ['path/to/images_list.txt', 'output_dir'] 
path_and_save_dir = ["path/to/images", "caption_backups"]

# Query prompt to use for captioning
query_prompt = 'Provide a very detailed description for this image.'

# How often to save of a backup of the captions
backup_iter=20

# Load images
images_list = load_images_and_parquet(path_and_save_dir, verbose=True)
print("Processing", len(images_list), "images")

# Initialize or load dataframe
output_file = os.path.join(path_and_save_dir[1], "output.parquet")
if os.path.exists(output_file):
    df = pd.read_parquet(output_file)
else:
    df = pd.DataFrame(columns=['image_name', 'hash', 'short_caption', 'long_caption', 'resolution'])

# Run model and caption images
run_model(model=model, tokenizer=tokenizer, df=df, images_list=images_list, backup_dir=path_and_save_dir[1], query_prompt=query_prompt, backup_iter=backup_iter)
