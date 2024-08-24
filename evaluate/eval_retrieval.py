import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor

from ..trainer import build_model

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]
    
    
class ImageDataset(Dataset):
    def __init__(self, images,  embedding_type = 'numpy'):
        self.imgs = images
        self.embedding_type = embedding_type
        
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        file_name = self.imgs[idx]
        if self.embedding_type == 'numpy':
            file_name = file_name + '.npy'
            img = np.load(file_name)
            
        elif self.embedding_type == 'image_string':
            img = file_name
            
        elif self.embedding_type == 'image':
            img = Image.open(file_name)
            
        file_name = os.path.basename(file_name).split('.')[0]
        return img, file_name


def get_dataset(directory = 'data/evaluate/imagenet'):
    
    # Not clarify how the data is stored, but we can assume that the data is stored in parquet format
    
    # Find the parquet file in the directory
    for file in os.listdir(directory):
        if file.endswith('.parquet'):
            image_text = pd.read_parquet(os.path.join(directory, file))
            
            # Test the original language
            # image_text.drop(columns=['caption'], inplace=True)
            # image_text.rename(columns={'original': 'caption'}, inplace=True)
            break
    
    image_text['image_id'] = pd.factorize(image_text['image'])[0]
    image_text['text_id'] = pd.factorize(image_text['caption'])[0]
    
    text_df = image_text[['text_id', 'caption']].drop_duplicates()
    image_df = image_text[['image_id', 'image']].drop_duplicates()
    
    text_df.sort_values('text_id', inplace=True)
    image_df.sort_values('image_id', inplace=True)

    image_df['image'] = image_df['image'].apply(lambda x: os.path.join(directory, 'images', x))

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


def save_embedding(file_name, embedding, path):
    np.save(f'{path}/{file_name}.npy', embedding)

# Multithread
def save_embeddings_in_parallel(file_names, embedding_batch, path):

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(save_embedding, file_name, embedding_batch[j], path)
            for j, file_name in enumerate(file_names)
        ]
        # Optionally wait for all threads to complete
        for future in futures:
            future.result()               
   
from multiprocessing import Process, Queue, cpu_count
import multiprocessing as mp        
# Function to save embeddings (consumer)

def save_embeddings_in_parallel2(file_names, embedding_batch, path):

    # with ThreadPoolExecutor() as executor:
    #     futures = [
    #         executor.submit(save_embedding, file_name, embedding_batch[j], path)
    #         for j, file_name in enumerate(file_names)
    #     ]
    #     # Optionally wait for all threads to complete
    #     for future in futures:
    #         future.result()    
    with mp.Pool(cpu_count()-1) as p:
        p.starmap(save_embedding, [(file_names[i], embedding_batch[i], path) for i in range(len(file_names))])

# Multiprocess multithread
def save_embedding_consumer2(queue, path):
    while True:
        file_names, embedding_batch = queue.get()
        if file_names is None:
            break  # Exit when a sentinel value (None) is received
        save_embeddings_in_parallel(file_names, embedding_batch, path)
        
# Multiprocess single thread
def save_embedding_consumer(queue, path):
    while True:
        file_name, embedding = queue.get()
        if file_name is None:
            break  # Exit when a sentinel value (None) is received
        np.save(f'{path}/{file_name}.npy', embedding.cpu().detach().numpy())

# Producer function that encodes images (main process)
def process_images_and_save(dataloader, model, path, num_workers=None):
    queue = Queue()

    # Determine number of workers (consumers) based on CPU cores if not provided
    if num_workers is None:
        num_workers = cpu_count()  # Use all available cores

    # Start consumer processes
    consumer_processes = []
    for _ in range(num_workers):

        p = Process(target=save_embedding_consumer, args=(queue, path))
        p.start()
        consumer_processes.append(p)

    # Producer loop (encoding and putting into the queue)
    # Cache the embeddings 
   
    i = 0
    for images, file_names in tqdm(dataloader):
        embedding_batch = model.encode_image(images)
        for j, file_name in enumerate(file_names):
            queue.put((file_name, embedding_batch[j]))
        # queue.put((file_names, embedding_batch))  # Put embeddings in queue

    # Signal the consumer processes to exit
    for _ in range(num_workers):
        queue.put((None, None))  # Sentinel value to signal the consumers to stop

    # Wait for all consumers to finish
    for p in consumer_processes:
        p.join()
        
        
        

    
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
            
        if hasattr(self.model, 'setup_training'):
            self.model.setup_training(train_vision=False, train_text=False, device=self.device)
        else:
            self.model.to(self.device)
        
        self.eval_args = eval_args
        self.is_embedding = False
        
        print("=============Loading dataset=============")
        self.dataset = get_dataset(eval_args['dataset'])
        
    def _encode_text(self, texts):
        print("=============Encoding texts=============")
        
        dataset = TextDataset(texts)
        dataloader = DataLoader(dataset, batch_size=self.eval_args['batch_size'], num_workers=self.eval_args['num_workers'], shuffle=False)
        
        # Encode text by batch
        i = 0
        with torch.no_grad():
            for texts in tqdm(dataloader):

                text_batch = self.model.encode_text(texts).cpu().detach().numpy() # (batch, embedding_dim), convert to numpy
                if i == 0:
                    text_embeddings = text_batch
                else:
                    text_embeddings = np.concatenate([text_embeddings, text_batch], axis=0)
                i+=1
        return text_embeddings
    
    @staticmethod
    def open_image(image_path):
        return Image.open(image_path)
    
    def _encode_image(self, images):
        print("=============Encoding images=============")
        
        dataset = ImageDataset(images, 'image_string')
        dataloader = DataLoader(dataset, batch_size=self.eval_args['batch_size'], num_workers=self.eval_args['num_workers'], shuffle=False)
        
        # Encode image by batch
        i = 0
        with torch.no_grad():
            for images, ids in tqdm(dataloader):
                # Open image

                image_batch = self.model.encode_image(images).cpu().detach().numpy() # (batch, embedding_dim), convert to numpy
                if i == 0:
                    image_embeddings = image_batch
                else:
                    image_embeddings = np.concatenate([image_embeddings, image_batch], axis=0)
                i+=1
        return image_embeddings
    
    # For pre-embedding images and save the embeddings
    def preembed_image(self, path = "data/embeddings", embedding_type = 'numpy', num_workers = 18):

        if not os.path.exists(path):
            os.makedirs(path)
        
        batch_size = self.eval_args['batch_size']
        dataset = ImageDataset(self.dataset['images']['image'].values, 'image_string')
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=self.eval_args['num_workers'], shuffle=False)
        
        embedding_batchs = None
        file_namess = []
        
        with torch.no_grad():
            # process_images_and_save(dataloader, self.model, path, num_workers=num_workers)
            for images, file_names in tqdm(dataloader):
                embedding_batch = self.model.encode_image(images).cpu().detach().numpy()
                # save_embeddings_in_parallel(file_names, embedding_batch, path)
                if embedding_batchs is None:
                    embedding_batchs = embedding_batch
                else:
                    embedding_batchs = np.concatenate([embedding_batchs, embedding_batch], axis=0)
                file_namess += file_names
        np.save(f'image_embeddings.npy', embedding_batchs)
        pd.DataFrame({"filename":file_namess}).to_csv('image_names.csv', index=False)
        # save_embeddings_in_parallel2(file_namess, embedding_batchs, path)
     

    def _evaluate(self, top_k = 5):
        if not self.is_embedding:
            self.image_embeddings = self._encode_image(self.dataset['images']['image'].values)
            self.text_embeddings = self._encode_text(self.dataset['texts']['caption'].values)
            self.is_embedding = True
            
        img_text_scores, text_img_scores = get_top_matches(self.image_embeddings, self.text_embeddings, top_k)
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
            return set(self.dataset['image_text'][self.dataset['image_text']['text_id'] == id]["image_id"])
        else:  # image
            return set(self.dataset['image_text'][self.dataset['image_text']['image_id'] == id]['text_id'])
    
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
            text_img_recall_k += len(relevant_images & retrieved_images_k) / top_k
            text_img_recall_1 += 1 if text_img[i,0] in relevant_images else 0
            
        # Retrieval for image to text
        for i in range(len_img):
            relevant_texts = self.get_relevant_items(i, 'image')
            retrieved_texts_k = set(img_text[i,:top_k])
            img_text_recall_k += len(relevant_texts & retrieved_texts_k) / top_k
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

        
            
        
        
        
    