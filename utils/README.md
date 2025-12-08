For translation, remember to clone the llm repo and set up the Gemini API key.


Rule for translation:

1. Use the Gemini LLM to translate image captions from English to Vietnamese.

2. Use the `translate.py` script to process the captions in batches.

3. Store the translated captions in JSONL format for easy retrieval and analysis.

4. Use the `check_translation_result.py` script to verify the translations progress, and save the results to a JSONL file.

5. Use the `merge_translate.py` script to combine the original captions with their translations into a single Parquet file for further use.

```
python embed_image.py --dataset_root ../../data/cc12m-raw --output_root ../../data/cc12m-siglip-b224 --model vit_base_patch16_siglip_224 --batch_size 256 --folder_start 651 --folder_end 950 --num_workers 8
```



```
python embed_image.py --model vit_base_patch16_clip_224.dfn2b --dataset_root ../../data/cc12m-raw --output_root ../../data/cc12m-clip-b224 --batch_size 512 --folder_start 0 --folder_end 150 --num_workers 8
```