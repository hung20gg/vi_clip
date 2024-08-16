File structure

```
\trainer
	dataloader.py
	trainer.py
\model
	base_model.py
	crosslingual.py
	lossfn.py
	...
\evaluate
preprocess.py
args.py
```

### Clone the repo and import

**Import models:**

```
from vi_clip.model import CLIP, SigLIP, LiT, SigLiT, CrossLingual, mCLIP, BaselineCLIP, BaselineSigLIP
from vi_clip.args import model_args

model = SigLiT(**model_args)
```

**Import trainers**

```
from vi_clip.trainer import Trainer, CrossLingualTrainer, ddp_train
from vi_clip.args import training_args, model_args

# Training with single GPU or DataParallel
training_args['train_type'] = 'single' # single GPU
training_args['train_type'] = 'dp' # DataParallel

trainer = Trainer(model_args, training_args)
trainer.train()

# Training with Distributed Data Parallel
training_args['train_type'] = 'ddp'
ddp_train(model_args, training_args)
```

### Model description

Every model has these attributes:

- text_model
- tokenizer
- vision_model
- processor

And these methods:

- encode_image() str, list[str] (for image dir), Image, list[Image], np.ndarray, torch.Tensor
- encode_text() str, list[str]
- forward(images, texts) same as those 2

Need changing:

- Only include text encoder
- Adding text projection layer (nn.Linear) even if both ViT and BERT embedding share the same dimension.
- Implementation for freezing BERT and only train projection layer in some early epoch + different learning rate.
- Test the `Evaluate` class
- Pre-embedding, upload and download scripts (file must be in some order idk)
- Only change the embedding layer with new vocab (which dataset to change the vocab ??)

### Download the dataset

import the `download_and_extract_batches` from `vi_clip.preprocess`, passing hf repo and local dir, it will download the repo (include image) in this structure

(These are some bugs at the moment, so hehe)

```
\dataset_name
	dataset_caption.parquet
	\image
		1.jpg
		2.jpg
```
**Data format**
The parquet file should be like this

| image_id | image | text_id | caption |
|----------|--------|--------|----------|
| 000001 | name.jpg| 005| giám đốc |


