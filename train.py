from args import training_args, model_args, eval_args, parse_to_train_model_eval_args
from trainer import Trainer, CrossLingualTrainer, mCLIPTrainer
from evaluate.eval_retrieval import EvaluateModel
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Training Script')
    
    # Training arguments
    parser.add_argument('--lr', type=float, default=training_args['lr'], help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=training_args['weight_decay'], help='Weight decay')
    parser.add_argument('--epochs', type=int, default=training_args['epochs'], help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=training_args['batch_size'], help='Batch size')
    parser.add_argument('--scheduler', type=str, default=training_args['scheduler'], help='Scheduler type')
    parser.add_argument('--warmup_steps', type=int, default=training_args['warmup_steps'], help='Number of warmup steps')
    parser.add_argument('--peak_lr', type=float, default=training_args['peak_lr'], help='Peak learning rate')
    parser.add_argument('--num_workers', type=int, default=training_args['num_workers'], help='Number of workers')
    # parser.add_argument('--training_objective', type=str, default=training_args['training_objective'], help='Training objective')
    parser.add_argument('--dataset', nargs='+', default=training_args['dataset'], help='Datasets')
    
    # Model arguments
    parser.add_argument('--text_model', type=str, default=model_args['text_model'], help='Text encoder model')
    parser.add_argument('--vision_model', type=str, default=model_args['text_model'], help='Image encoder model')
    parser.add_argument('--model_type', type=str, default=model_args['model_type'], help='Model type')
    
    # Evaluation arguments
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