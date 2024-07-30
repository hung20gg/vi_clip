import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from ..model import build_model

def get_dataset(name):
    if name == 'mscoco_vi':
        df = pd.read_csv('data/evaluate/mscoco_vi.csv')
    elif name == 'imagenet1k_vi':
        df = pd.read_csv('data/evaluate/imagenet1k_vi.csv')
    elif name == 'imagenet21k_vi':
        df = pd.read_csv('data/evaluate/imagenet21k_vi.csv')
    else:
        raise ValueError('Dataset not found')
    
    df['image_id'] = None 
    df['text_id'] = None
    
    return {
        'images': df['image_path'].values,
        'texts': df['label'].values,
        'image_id': df['image_id'].values,
        'text_id': df['text_id'].values
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
    
class EvaluateRAG:
    def __init__(self, model_args, eval_args, device = None):
        
        if device is not None:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        if isinstance(model_args, dict):
            self.model = build_model(model_args)
            if model_args['checkpoint'] is not None:
                self.model.load_checkpoint(torch.load(model_args['checkpoint']))
        else:
            self.model = model_args
            
        self.model.to(self.device)
        
        self.eval_args = eval_args
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
    
    def _encode_image(self, images):
        for i in range(0, len(images), self.eval_args['batch_size']):
            image_batch = images[i:i+self.eval_args['batch_size']]
            image_batch = self.model.encode_image(image_batch)
            if i == 0:
                image_embeddings = image_batch
            else:
                image_embeddings = np.concatenate([image_embeddings, image_batch], axis=0)
        return image_embeddings
    
    def _evaluate(self, top_k = 5):
        image_embeddings = self._encode_image(self.dataset['images'])
        text_embeddings = self._encode_text(self.dataset['texts'])
        
        img_text_scores, text_img_scores = get_top_matches(image_embeddings, text_embeddings, top_k)
        return img_text_scores, text_img_scores
        # Calculate Acc@K k = 1, 5
        
    def zero_shot_classification(self):
        return 0
    
    def get_relevant_items(self, id, id_type):
        if id_type == 'text':
            return set(self.dataset['image_id'][self.dataset['text_id'] == id])
        else:  # image
            return set(self.dataset['text_id'][self.dataset['image_id'] == id])
    
    def retrieval(self, top_k = 5):
        img_text, text_img = self._evaluate()
        
        text_img_recall_1 = 0
        text_img_recall_k = 0
        img_text_recall_1 = 0
        img_text_recall_k = 0
        
        len_text = len(self.dataset['text_id'].unique())
        len_img = len(self.dataset['image_id'].unique())
        
        for i in range(len_text):
            relevant_images = self.get_relevant_items(i, 'text')
            retrieved_images_k = set(text_img[i,:top_k])
            text_img_recall_k += len(relevant_images & retrieved_images_k) / min(len(retrieved_images_k),top_k)
            text_img_recall_1 += 1 if text_img[i,0] in relevant_images else 0
            
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

        
            
        
        
        
    