from args import training_args, model_args, eval_args, parse_to_train_model_eval_args
from trainer import Trainer
from evaluate.eval_retrieval import EvaluateModel
import argparse

###############################################
#
# This file is used if using bash command to 
# train the model. But there are problems with 
# import libraries.
#
################################################

def parse_args():
    parser = argparse.ArgumentParser(description='Training Script')
    
    # Training arguments
    parser.add_argument('--train_name', type=str, default=training_args['train_name'], help='Name of the training experiment')
    parser.add_argument('--wandb_project', type=str, default=training_args.get('wandb_project', None), help='WandB project name')
    parser.add_argument('--train_type', type=str, default=training_args['train_type'], help='Training type')
    parser.add_argument('--mixed_precision', type=bool, default=training_args['mixed_precision'], help='Mixed precision training')
    parser.add_argument('--device', type=str, default=training_args['device'], help='Device')
    parser.add_argument('--lr', type=float, default=training_args['lr'], help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=training_args['weight_decay'], help='Weight decay')
    parser.add_argument('--epochs', type=int, default=training_args['epochs'], help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=training_args['batch_size'], help='Batch size')
    parser.add_argument('--scheduler', type=str, default=training_args['scheduler'], help='Scheduler type')
    parser.add_argument('--warmup_steps', type=int, default=training_args['warmup_steps'], help='Number of warmup steps')
    parser.add_argument('--peak_lr', type=float, default=training_args['peak_lr'], help='Peak learning rate')
    parser.add_argument('--initial_lr', type=float, default=training_args['initial_lr'], help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=training_args['num_workers'], help='Number of workers')
    parser.add_argument('--dataset', nargs='+', default=training_args['dataset'], help='Datasets')
    parser.add_argument('--save_dir', type=str, default=training_args['save_dir'], help='Save directory')
    parser.add_argument('--beta2', type=float, default=training_args['beta2'], help='Beta2 of AdamW')
    parser.add_argument('--data_type', type=str, default=training_args.get('data_type', 'images'), help='Type of data: images or numpy')
    parser.add_argument('--accelerate', type=bool, default=training_args['accelerate'], help='Accelerate training on Ampere GPUs')
    parser.add_argument('--train_projection_only', type=bool, default=training_args.get('train_projection_only'), help='Evaluate every n iterations')
    parser.add_argument('--text_projection_iters', type=bool, default=training_args.get('text_projection_iters'), help='Evaluate every n iterations')
    parser.add_argument('--evaluate_every', type=int, default=training_args.get('evaluate_every', 200), help='Evaluate every n iterations')
    parser.add_argument('--log_every', type=int, default=training_args.get('log_every', 20), help='Trim dataset for debugging')
    parser.add_argument('--hf_repo_name', type=str, default=training_args.get('hf_repo_name', None), help='Hugging Face repository name')

    # Model arguments
    parser.add_argument('--text_model', type=str, default=model_args.get('text_model'), help='Text encoder model')
    parser.add_argument('--vision_model', type=str, default=model_args.get('vision_model'), help='Image encoder model')
    parser.add_argument('--clip_model', type=str, default=model_args.get('clip_model'), help='CLIP model')
    parser.add_argument('--model_type', type=str, default=model_args.get('model_type'), help='Model type')
    parser.add_argument('--max_length', type=int, default=model_args.get('max_length'), help='Maximum length')
    parser.add_argument('--pretrain', type=bool, default=model_args.get('pretrain'), help='Pretrain model')
    parser.add_argument('--force_text_projection', type=bool, default=model_args.get('force_text_projection'), help='Force text projection')
    
    # Evaluation arguments
    parser.add_argument('--is_eval', type=bool, default=False, help='Evaluate model')
    parser.add_argument('--eval_batch_size', type=int, default=eval_args.get('batch_size', 32), help='Evaluation batch size')
    parser.add_argument('--eval_num_workers', type=int, default=eval_args.get('num_workers', 4), help='Number of workers for evaluation')
    parser.add_argument('--eval_dataset', type=str, default=eval_args.get('dataset', 'default_dataset'), help='Evaluation dataset')
    
    return parser.parse_args()

args = parse_args() 
training_args, model_args, eval_args = parse_to_train_model_eval_args(args)



trainer = Trainer(model_args, training_args)
    
print('Training model:', model_args)
print('Training arguments:', training_args)
    
# trainer.train()

trainer.train()

# evaluate = EvaluateModel(trainer.model, eval_args)
# evaluate.zero_shot_classification()
# evaluate.retrieval()
