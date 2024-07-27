import os
import re
import torch
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def prepare_inputs(user_message, tokenizer):
    """
    Prepares inputs for the model by formatting user messages with predefined prompts and responses.

    Args:
        user_message (str): The user's message to be summarized.
        tokenizer (Tokenizer): Tokenizer for encoding the messages.

    Returns:
        torch.Tensor: Tokenized and formatted inputs suitable for model processing.
    """
    # Prompts and responses for showing the model what is expected of it
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Summarize the image captions below in a single sentence. The summary should objectively only mention elements, objects and subjects present in the caption. You must present the caption between tags [caption][/caption]. Do not say 'the caption'. Here goes a caption you need to summarize: 'The image depicts a young girl, possibly in her early teens, sitting on the steps of a grand, multi-storied building with a distinct architectural style. She is engrossed in reading a book. The girl is dressed in a green uniform with a badge on her left chest. Behind her, the building is surrounded by lush greenery, and in the distance, majestic mountains rise under a clear blue sky. The artwork appears to be hand-drawn, possibly using watercolors, and has a serene, tranquil ambiance.'"},
        {"role": "assistant", "content": "[caption]A young girl in a green uniform sits reading a book on the steps of a grand building with a mountainous backdrop.[/caption]"},
        {"role": "user", "content": "Now, summarize the following: 'This image showcases a group of multiple human figures intertwined and stacked on top of each other on the edge of a high building. The figures are nude, and their intertwined poses create a complex and intricate formation. The backdrop is a sprawling cityscape with numerous skyscrapers, roads, and other urban structures. The lighting suggests it's either early morning or late afternoon, casting a soft glow over the scene.'"},
        {"role": "assistant", "content": "[caption]Multiple human figures intertwine and embrace on a rooftop, with a sprawling cityscape below them, bathed in the soft glow of the setting or rising sun.[/caption]"},
        {"role": "user", "content": "Now, summarize the following:" + user_message}
    ]
    return tokenizer.apply_chat_template(messages, return_tensors="pt")


def summarize_text(long_caption, model, tokenizer):
    """
    Generates a summary of a long caption using the provided model.

    Args:
        long_caption (str): The long caption to be summarized.
        model (nn.Module): The summarization model.
        tokenizer (Tokenizer): Tokenizer for encoding the captions.

    Returns:
        str: The generated summary of the caption.
    """
    model_inputs = prepare_inputs(long_caption, tokenizer).to("cuda")
    # Suppress annoying messages with: 'pad_token_id=tokenizer.eos_token_id'
    generated_ids = model.generate(model_inputs, max_new_tokens=100, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    output = tokenizer.batch_decode(generated_ids)[0].strip()
    return output


def generate_short_caption(caption, model, tokenizer):
    """
    Generates a short caption from a given caption using the provided model.

    Args:
        caption (str): The caption to be summarized.
        model (nn.Module): The summarization model.
        tokenizer (Tokenizer): Tokenizer for encoding the captions.

    Returns:
        str: The generated short caption.
    """
    summarization = '''Now, summarize the following: ''' + caption
    not_ok = True
    while not_ok:
        short_caption = summarize_text(summarization, model, tokenizer)
        pattern = r'\[caption\](.*?)\[/caption\]'
        matches = re.findall(pattern, short_caption, re.DOTALL)
        try:
            short_caption = matches[3] if len(matches) >= 1 else "No content found"
            not_ok = False
        except:
            continue
        if short_caption=='' or short_caption==' ':
            continue

    return short_caption


# Load Parquet file with long captions and empty short captions
input_parquet_filename = "captions.parquet"
output_filename = "captions_with_short_captions.parquet"
model_repo = "meta-llama/Meta-Llama-3.1-8B-Instruct"
hf_auth = "<auth_token>"


if os.path.exists(output_filename):
    df = pd.read_parquet(output_filename)
else:
    df = pd.read_parquet(input_parquet_filename)

num_missing_short_captions = df[df['short_caption'].isin([None, '', ' ', 'No content found'])].shape[0]
print("Number of missing short captions:", num_missing_short_captions)


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_repo, token=hf_auth)
model = AutoModelForCausalLM.from_pretrained(
    model_repo,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    load_in_4bit=True,
    trust_remote_code=True,
    token=hf_auth
).eval()


# Process each row	
ite = 0
for index, row in tqdm(df.iterrows(), total=len(df.values.tolist())):
    if pd.isna(row['short_caption']) or row['short_caption'] in ['No content found', '', ' ']:
        if pd.notna(row['long_caption']) and row['long_caption'] != '':
            short_caption = generate_short_caption(row['long_caption'], model, tokenizer)
            df.at[index, 'short_caption'] = short_caption
            ite += 1
            print(short_caption)
            # Save the updated DataFrame every 10 iterations
            if ite % 10 == 0:
                df.to_parquet(output_filename)
df.to_parquet(output_filename)
