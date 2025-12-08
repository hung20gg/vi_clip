from llm import Gemini
import json

prefix = "0151_0200"

llm = Gemini(model_name="gemini-2.0-flash-lite")

batch_ids = llm.recall_local_batch(prefix)
print("Recalled batch IDs:")
print(batch_ids)

results = llm.get_successful_messages(batch_ids)

with open(f"../../data/cc12m-raw/translations_{prefix}.jsonl", 'w', encoding='utf-8') as f:
    for result in results:
        obj = {
            'image_id': result['key'],
            'translated_caption': result['text']
        }
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")