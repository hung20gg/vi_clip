from llm import Gemini
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import time

PROMPT = "Translate the following English text to Vietnamese. Only return the translation. The original text is an image description, so you don't translate any additional context.\n\nEnglish: {text}\nVietnamese:"


llm = Gemini(model_name="gemini-2.0-flash-lite")

def _translate_text(llm_wrapper: Gemini, text: str) -> str:
    prompt = PROMPT.format(text=text, temperature=0.6)
    response = llm_wrapper(prompt)
    return response

def _translate_text_batch(llm_wrapper: Gemini, texts: list[str], keys: list[str], prefix=None) -> None:
    prompts = [PROMPT.format(text=text, temperature=0.6) for text in texts]
    
    llm_wrapper.batch_call(prompts, key_list=keys, example_per_batch=1000, prefix=prefix)


def process_translation_by_pd_row(llm_wrapper, row, save_path=None):

    caption = row['caption']

    if len(caption.split()) > 60:
        return  # Skip long captions
    
    translated_text = _translate_text(llm_wrapper, row['caption'])

    id_ = row['image_id']

    if save_path is not None:
        obj = {
            'image_id': id_,
            'translated_caption': translated_text
        }

        with open(save_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")



def translate_dataframe(df, llm_wrapper=None, save_path=None, max_workers=4):
    """
    Translate captions in a DataFrame using multithreading.
    
    Args:
        df: DataFrame with 'caption' and 'image_id' columns
        llm_wrapper: LLM wrapper instance (defaults to global llm)
        save_path: Path to save translations
        max_workers: Number of threads to use
    """
    if llm_wrapper is None:
        llm_wrapper = llm

    if 'translated_caption' in df.columns:
        df = df[df['translated_caption'].isnull() | (df['translated_caption'] == '')]
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for _, row in df.iterrows():
            future = executor.submit(
                process_translation_by_pd_row,
                llm_wrapper,
                row,
                save_path
            )
            futures.append(future)
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Translating"):
            try:
                future.result()
            except Exception as e:
                print(f"Error translating row: {e}")


def translate_dataframe_in_batches(df, llm_wrapper=None, prefix=None, batch_size=1000, max_workers=4):
    """
    Translate captions in a DataFrame using multithreading and batch processing.
    
    Args:
        df: DataFrame with 'caption' and 'image_id' columns
        llm_wrapper: LLM wrapper instance (defaults to global llm)
        save_path: Path to save translations
        batch_size: Number of captions per batch
        max_workers: Number of threads to use
    """
    if llm_wrapper is None:
        llm_wrapper = llm

    if 'translated_caption' in df.columns:
        df = df[df['translated_caption'].isnull() | (df['translated_caption'] == '')]

    captions = df['caption'].tolist()
    image_ids = df['image_id'].tolist()

    print(f"Total captions to translate: {len(captions)}")

    for i in range(0, len(captions), batch_size):
        batch_captions = captions[i:i + batch_size]
        batch_image_ids = image_ids[i:i + batch_size]

        _translate_text_batch(llm_wrapper, batch_captions, batch_image_ids, prefix=prefix)

        time.sleep(10)  # Optional: to avoid rate limits

if __name__ == "__main__":
    file_name = "0151_0200"
    parquet_path = f"../../data/cc12m-raw/captions_{file_name}.parquet"
    save_path = f"../../data/cc12m-raw/translations_{file_name}.jsonl"
    df = pd.read_parquet(parquet_path)
    translate_dataframe_in_batches(df, prefix=file_name, batch_size=1000, max_workers=4)