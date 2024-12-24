import os
from utils import get_args_vae

# Parse input augments
args = get_args_vae()

# Set PyTorch to use only the specified GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpu))

# Make project directory if not exist
if not os.path.exists(args.log_dir):
    os.makedirs(args.log_dir)

from trainer import SurfVAETrainer
from datasets.sxd import SurfData
from trainer import EdgeVAETrainer
from datasets.sxd import EdgeData

def run(args):
    print('Args:', args)
    
    # Initialize dataset loader and trainer
    if args.option == 'surface':
        train_dataset = SurfData(
            args.data, args.train_list, data_fields=args.data_fields, 
            validate=False, aug=args.data_aug, chunk_size=args.chunksize)
        val_dataset = SurfData(
            args.data, args.val_list, data_fields=args.data_fields, 
            validate=True, aug=False, chunk_size=args.chunksize)
        vae = SurfVAETrainer(args, train_dataset, val_dataset)
    else:
        assert args.option == 'edge', 'please choose between surface or edge'
        train_dataset = EdgeData(args.data, args.train_list, validate=False, aug=args.data_aug)
        val_dataset = EdgeData(args.data, args.val_list, validate=True, aug=False)
        vae = EdgeVAETrainer(args, train_dataset, val_dataset)

    # Main training loop
    print('Start training...')
    
    for _ in range(args.train_nepoch):  

        # Train for one epoch
        vae.train_one_epoch()

        # Evaluate model performance on validation set
        if vae.epoch % args.test_nepoch == 0:
            vae.test_val()

        # save model
        if vae.epoch % args.save_nepoch == 0:
            vae.save_model()
    return
           

if __name__ == "__main__":
    run(args)