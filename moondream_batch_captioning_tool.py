import torch
import pandas as pd
from PIL import Image, ImageFile
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
from hashlib import md5
import os
from pathlib import Path
from itertools import chain
from tqdm import tqdm
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
        # from numba import jit

        # @jit(nopython=True, cache=True) # Remove if not using numba JIT
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


def load_file_paths_from_directory(directory_path, allowed_extensions=['.jpg', '.jpeg', '.png']):
    """
    Load a list of file paths from a directory with specified allowed extensions.

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
            images_list += load_file_paths_from_directory(s, allowed_extensions=allowed_extensions)

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


def run_model(model, tokenizer, df, images_list, backup_dir, query_prompt, backup_iter=20, failure_threshold=70.0, batch_size=8, max_attempts=5, verbose=True, device=torch.device("cuda")):
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
        failure_threshold (float, optional): Threshold for caption repeat detection. Set to None to
            disable. Defaults to 70.0.
        batch_size (int, optional): The number of images per batch. Defaults to 8.
        max_attempts (int, optional): The number of times to attempt recaptioning images for failed
            captions. This parameter is only in play if failure_threshold is not None. Defaults to 5.
        verbose (bool, optional): Whether or not to print additional information like number of
            duplicates skipped. Default: True
        device (torch.device, optional): The PyTorch device to use for the model and images.
            Default: torch.device("cuda")
    """
    max_attempts = max_attempts if failure_threshold else 0
    column_length = len(df['image_name'])
    existing_hash_dict = {hash_str: hash_str for hash_str in hash_list}
    skip_total = 0
    lrs = LRSCalculationWorker()

    # Process each batch of images
    for index in tqdm(range(0, len(images_list), batch_size), desc="Processing images"):
        batch_set = images_list[index:index+batch_size]
        batch_images = []
        batch_image_paths = []
        for image_path in batch_set:
            if not image_path:
                continue

            try:
                batch_images += [Image.open(image_path)]
                batch_image_paths += [image_path]
            except Exception as e:
                print(f"Failed to load image {image_path}: {e}")
                continue

            if not batch_images:
                continue

        if len(batch_images) == 0:
            print("Image hashes already exists in saved captions, skipping...")
            skip_total += len(batch_set)
            continue

        image_hashes = [md5(image.tobytes()).hexdigest() for image in batch_images]

        # Filter out duplicate images based on hash
        existing_hash_set = set(df['hash'])
        filtered_batch_images = [(img, img_hash) for img, img_hash in zip(batch_images, image_hashes) if img_hash not in existing_hash_set]
        batch_images = [b_img[0] for b_img in filtered_batch_images]
        image_hashes = [b_img[1] for b_img in filtered_batch_images]
        if len(batch_images) > len(batch_set):
            skip_total += len(batch_set) - len(batch_images)

        if verbose and len(batch_images) < len(batch_set):
             print("Skipped",  len(batch_set) - len(batch_images), "duplicate images")
        if len(batch_images) == 0:
             skip_total += len(batch_set)
             continue

        # Encode batch of images
        with torch.no_grad():
            captions = model.batch_answer(batch_images, [query_prompt] * len(batch_images), tokenizer)

        # Process captions
        captions = [caption.replace("</s>", "").strip() for caption in captions]

        failed_captions = [[lrs.find_lrs(cap[-128:])[1], im] for cap, im in zip(captions, batch_images)]
        failed_captions = [[i] + fc for i, fc in enumerate(failed_captions)]  # Add index
        f_batch = [fc for fc in failed_captions if fc[1] > failure_threshold]
        if max_attempts > 0 and len(f_batch) > 0:
            count = 0
            for _ in range(max_attempts):
                if verbose:
                    print(f'Got bad captions, retrying...', str(count + 1) + " / " + str(max_attempts))
                    print("\n".join([captions[idx[0]] for idx in f_batch]))
                count += 1
                with torch.no_grad():
                    gen_args = generate_random_gen_kwargs()
                    caps = model.batch_answer([f[2] for f in f_batch], [query_prompt] * len(f_batch), tokenizer, **gen_args)
                for i in range(len(f_batch)):
                    idx = f_batch[i][0]
                    # Replace caption values with corrected captions
                    captions[idx] = caps[i]
                failed_captions = [[lrs.find_lrs(cap[-128:])[1], im] for cap, im in zip(captions, batch_images)]
                failed_captions = [[i] + fc for i, fc in enumerate(failed_captions)]
                f_batch = [fc for fc in failed_captions if fc[1] > 70.0]
                if len(f_batch) < 1:
                     break


        # Create new rows and concatenate them with the existing dataframe
        new_rows = pd.DataFrame({'image_name': batch_image_paths,
                                 'hash': image_hashes,
                                 'short_caption': [''] * len(batch_image_paths),
                                 'long_caption': captions,
                                 'resolution': resolutions})
        df = pd.concat([df, new_rows], ignore_index=True)
    
        index = ((index - skip_total) + column_length) + len(captions)
        if index % backup_iter == 0 and index > 0:
            backup_file = f'output_backup_{index}.parquet'
            df.to_parquet(os.path.join(backup_dir, backup_file))
    
        df.to_parquet(output_file)



# Initialize the tokenizer and model
model_id = "vikhyatk/moondream2"
revision = "2024-05-20"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, revision=revision, torch_dtype=torch.float16
).cuda()

tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
print("Model loaded")

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
