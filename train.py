from args import training_args, model_args, eval_args, parse_to_train_model_eval_args
from trainer import Trainer, CrossLingualTrainer, mCLIPTrainer
from evaluate.eval_retrieval import EvaluateModel
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training Script')
    
    # Training arguments
    parser.add_argument('--train_type', type=str, default=training_args['train_type'], help='Training type')
    parser.add_argument('--mix_precision', type=bool, default=training_args['mix_precision'], help='Mixed precision training')
    parser.add_argument('--device', type=str, default=training_args['device'], help='Device')
    parser.add_argument('--lr', type=float, default=training_args['lr'], help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=training_args['weight_decay'], help='Weight decay')
    parser.add_argument('--epochs', type=int, default=training_args['epochs'], help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=training_args['batch_size'], help='Batch size')
    parser.add_argument('--scheduler', type=str, default=training_args['scheduler'], help='Scheduler type')
    parser.add_argument('--warmup_steps', type=int, default=training_args['warmup_steps'], help='Number of warmup steps')
    parser.add_argument('--peak_lr', type=float, default=training_args['peak_lr'], help='Peak learning rate')
    parser.add_argument('--intial_lr', type=float, default=training_args['intial_lr'], help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=training_args['num_workers'], help='Number of workers')
    parser.add_argument('--dataset', nargs='+', default=training_args['dataset'], help='Datasets')
    parser.add_argument('--image_folder', type=str, default=training_args['image_folder'], help='Image folder')
    parser.add_argument('--save_dir', type=str, default=training_args['save_dir'], help='Save directory')
    parser.add_argument('--beta2', type=float, default=training_args['beta2'], help='Beta2 of AdamW')
    
    parser.add_argument('--accelerate', type=bool, default=training_args['accelerate'], help='Accelerate training on Ampere GPUs')
    
    # Model arguments
    parser.add_argument('--text_model', type=str, default=model_args['text_model'], help='Text encoder model')
    parser.add_argument('--vision_model', type=str, default=model_args['text_model'], help='Image encoder model')
    parser.add_argument('--clip_model', type=str, default=model_args['clip_model'], help='CLIP model')
    parser.add_argument('--model_type', type=str, default=model_args['model_type'], help='Model type')
    parser.add_argument('--max_length', type=int, default=model_args['max_length'], help='Maximum length')
    parser.add_argument('--pretrain', type=bool, default=model_args['pretrain'], help='Pretrain model')
    
    # Evaluation arguments
    parser.add_argument('--is_eval', type=bool, help='Evaluate model')
    parser.add_argument('--eval_batch_size', type=int, default=eval_args['batch_size'], help='Evaluation batch size')
    parser.add_argument('--eval_num_workers', type=int, default=eval_args['num_workers'], help='Number of workers for evaluation')
    parser.add_argument('--eval_dataset', nargs='+', default=eval_args['dataset'], help='Evaluation dataset')
    
    return parser.parse_args()

args = parse_args() 
training_args, model_args, eval_args = parse_to_train_model_eval_args(args)



# if model_args['model_type'] == 'crosslingual':
#     trainer = CrossLingualTrainer(model_args, training_args)
# elif model_args['model_type'] == 'mclip':
#     trainer = mCLIPTrainer(model_args, training_args)
# else:
#     trainer = Trainer(model_args, training_args)
    
print('Training model:', model_args)
print('Training arguments:', training_args)
    
# trainer.train()

# evaluate = EvaluateModel(trainer.model, eval_args)
# evaluate.zero_shot_classification()
# evaluate.retrieval()