import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from ..trainer import build_model

def get_dataset(directory = 'data/evaluate/imagenet'):
    
    # Not clarify how the data is stored, but we can assume that the data is stored in parquet format
    
    # Get 3 different datasets: image_text, image_text_id, image_id_text
    image_text = pd.read_parquet(f'{directory}/image_text.parquet')
    image_text['image_id'] = pd.factorize(image_text['image'])[0]
    image_text['text_id'] = pd.factorize(image_text['caption'])[0]
    
    text_df = image_text[['text_id', 'caption']].drop_duplicates()
    image_df = image_text[['image_id', 'image']].drop_duplicates()
    
    text_df.sort_values('text_id', inplace=True)
    image_df.sort_values('image_id', inplace=True)

    image_df['image'] = image_df['image'].apply(lambda x: f'{directory}/images/{x}')

    return {
        'images': image_df,
        'texts': text_df,
        'image_text': image_text,
    }

def get_top_matches(image_embeddings, text_embeddings, top_k = 5):
    """
    Calculate similarity scores between image embeddings and a text embedding,
    and return the top 5 highest matches.
    
    Args:
    image_embeddings (np.array): Image embeddings with shape (n_images, embedding_dim)
    text_embedding (np.array): Text embeddings with shape (n_texts, embedding_dim)
    image_ids (list): List of image IDs or names corresponding to the image embeddings
    
    Returns:
    
    """
    
    # Ensure inputs are numpy arrays
    image_embeddings = np.array(image_embeddings)
    text_embeddings = np.array(text_embeddings)
    
    # Normalize embeddings
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    
    # Calculate cosine similarities
    scores = np.dot(image_embeddings, text_embeddings.T) # (image,text)
    
    img_text_scores = np.argsort(scores, axis=1)[:, ::-top_k]
    text_img_scores = np.argsort(scores.T, axis=1)[:,::-top_k]
    
    return img_text_scores, text_img_scores
    
class EvaluateModel:
    def __init__(self, model_args, eval_args, device = None):
        
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
           
        # passing model, either through args or directly 
        print("=============Building model=============")
        if isinstance(model_args, dict):
            self.model = build_model(model_args)
            if model_args.get("checkpoint") is not None:
                self.model.load_checkpoint(torch.load(model_args['checkpoint']))
        else:
            self.model = model_args
            
        self.model.to(self.device)
        
        self.eval_args = eval_args
        
        print("=============Loading dataset=============")
        self.dataset = get_dataset(eval_args['dataset'])
        
    def _encode_text(self, texts):
        for i in range(0, len(texts), self.eval_args['batch_size']):
            text_batch = texts[i:i+self.eval_args['batch_size']]
            text_batch = self.model.encode_text(text_batch)
            if i == 0:
                text_embeddings = text_batch
            else:
                text_embeddings = np.concatenate([text_embeddings, text_batch], axis=0)
        return text_embeddings
    
    @staticmethod
    def open_image(image_path):
        return Image.open(image_path)
    
    def _encode_image(self, images):
        for i in range(0, len(images), self.eval_args['batch_size']):
            image_batch = images[i:i+self.eval_args['batch_size']]
            image_batch = [self.open_image(image) for image in image_batch]
            image_batch = self.model.encode_image(image_batch)
            if i == 0:
                image_embeddings = image_batch
            else:
                image_embeddings = np.concatenate([image_embeddings, image_batch], axis=0)
        return image_embeddings
    
    def _evaluate(self, top_k = 5):
        image_embeddings = self._encode_image(self.dataset['images']['image'].values)
        text_embeddings = self._encode_text(self.dataset['texts']['caption'].values)
        
        img_text_scores, text_img_scores = get_top_matches(image_embeddings, text_embeddings, top_k)
        return img_text_scores, text_img_scores

        
    def zero_shot_classification(self):
        # Evaluate zero-shot classification accuracy
        # Using the top 1 match image-text retrieval
        img_text_scores, text_img_scores = self._evaluate(top_k=1)
        image_text = self.dataset['image_text'].sort_values('image_id')
        
        check_order = np.arange(len(image_text))
        
        assert np.all(check_order == image_text['image_id'].values), 'Image order is not correct'
        labels = image_text['text_id'].values
        preds = img_text_scores[:,0] 
        return np.mean(labels == preds)
        
        
    
    def get_relevant_items(self, id, id_type):
        if id_type == 'text':
            return set(self.dataset['image_text'][self.dataset['image_text']['text_id'] == id])
        else:  # image
            return set(self.dataset['image_text'][self.dataset['image_text']['image_id'] == id])
    
    def retrieval(self, top_k = 5):
        img_text, text_img = self._evaluate()
        
        text_img_recall_1 = 0
        text_img_recall_k = 0
        img_text_recall_1 = 0
        img_text_recall_k = 0
        
        len_text = len(self.dataset['texts'])
        len_img = len(self.dataset['images'])
        
        # Retrieval for text to image
        for i in range(len_text):
            relevant_images = self.get_relevant_items(i, 'text')
            retrieved_images_k = set(text_img[i,:top_k])
            text_img_recall_k += len(relevant_images & retrieved_images_k) / min(len(retrieved_images_k),top_k)
            text_img_recall_1 += 1 if text_img[i,0] in relevant_images else 0
            
        # Retrieval for image to text
        for i in range(len_img):
            relevant_texts = self.get_relevant_items(i, 'image')
            retrieved_texts_k = set(img_text[i,:top_k])
            img_text_recall_k += len(relevant_texts & retrieved_texts_k) / min(len(retrieved_texts_k),top_k)
            img_text_recall_1 += 1 if img_text[i,0] in relevant_texts else 0
            
        text_img_recall_1 /= len_text
        text_img_recall_k /= len_text
        img_text_recall_1 /= len_img
        img_text_recall_k /= len_img
        
        return {
            'text_img_recall_1': text_img_recall_1,
            'text_img_recall_k': text_img_recall_k,
            'img_text_recall_1': img_text_recall_1,
            'img_text_recall_k': img_text_recall_k
        }
        
    def evaluate(self, task = 'retrieval', top_k = 5):
        print(f'Evaluating dataset {self.eval_args["dataset"]}, task: {task}')
        if task == 'retrieval':
            result = self.retrieval(top_k)
            print(f'Text-Image Recall@1: {result["text_img_recall_1"]}')
            print(f'Text-Image Recall@{top_k}: {result["text_img_recall_k"]}')
            print(f'Image-Text Recall@1: {result["img_text_recall_1"]}')
            print(f'Image-Text Recall@{top_k}: {result["img_text_recall_k"]}')
        
        elif task == 'zero_shot_classification':
            acc = self.zero_shot_classification()
            print(f'Zero-shot classification accuracy: {acc}')
        else:
            raise ValueError('Task not found')

        
            
        
        
        
    