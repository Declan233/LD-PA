import torch
from torch import optim
from tqdm import tqdm
import logging
import os
import numpy as np
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from dataloader import load_dataset
from utils import get_filename, plot_log
from engine import pairtrain_epoch
from network import MLP_SCA

no_cuda =False
cuda = not no_cuda and torch.cuda.is_available()
device = torch.device('cuda') if cuda else torch.device('cpu')


def run():
    parser = argparse.ArgumentParser(description="Arguments for LD-PA training.")
    parser.add_argument("-fp", "--file_path", type=str, default='./data', help="Path to data")
    parser.add_argument("--ref", type=str, default='ASCADv', help="Reference dataset")
    parser.add_argument("--target", type=str, default='ASCADf', help="Target dataset")
    parser.add_argument("--target_byte", type=int, default=0, choices=np.arange(0,16,1), help="Targeting byte")
    parser.add_argument("--target_dim", type=int, default=10000, help="Dimension of target traces.")
    parser.add_argument("--feature_dim", type=int, default=100, help="Dimension of features.")
    parser.add_argument("--window", type=int, default=None, help="Downsampling window.")
    parser.add_argument("-nr", "--nb_ref", type=int, default=200000, help="Number of reference traces")
    parser.add_argument("-nt", "--nb_tar", type=int, default=50000, help="Number of target traces for training")
    parser.add_argument("-ntv", "--nb_tar_valid", type=int, default=5000, help="Number of target traces for validation")
    parser.add_argument("--refidx", type=int, default=1, help="Specify which reference.")
    parser.add_argument("-lm", "--labeling_method", type=str, default='ID', choices=['ID', 'HW'], help="Labeling method")
    
    parser.add_argument("-lr", "--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("-e", "--epoch", type=int, default=50, help="Training epochs")
    parser.add_argument("-bs", "--batch_size", type=int, default=400, help="Batch size")
    parser.add_argument("--flag", type=str, default=None, help="Flags for signifying distinct experiments")
    
    parser.add_argument("--alpha", type=float, default=10, help="Balancing factor for LD-PA loss.")
    parser.add_argument('-rn',"--ref_noise", type=float, default=0, help="Addictive noise of reference.")
    parser.add_argument('-al',"--aug_level", type=float, default=0, help="Level of augmentation.")

    parser.add_argument("--earlystop", action='store_true', default=False, help="Enable early stopping.")

    args = parser.parse_args()
    # print(f"Command line arguments: {args}")

    ref_dataset = f'{args.file_path}/{args.ref}_{args.feature_dim}poi_{args.labeling_method}.h5'
    tar_dataset = f'{args.file_path}/{args.target}_nopoi_{args.window}window.h5'

    class_dim = 256 if args.labeling_method=='ID' else 9

    # network hyperparameters
    # generator_hp = None # random search hyperparameters when set to None
    generator_hp = {
        "nb_FC_layer": 3,
        "FC_neuron": [200,100,50],
        "activation": "leakyrelu"
    }
    classifier_hp = {
        "nb_FC_layer": 2,
        "FC_neuron": [100, 100],
        "activation": "elu"
    }

    # dataloader for training
    train_loader, valid_loader = load_dataset(ref_dataset=ref_dataset, 
                                        tar_dataset=tar_dataset, 
                                        num_ref=args.nb_ref, 
                                        num_tar=args.nb_tar, 
                                        num_tar_valid=args.nb_tar_valid,
                                        target_byte=args.target_byte, 
                                        labeling_method=args.labeling_method, 
                                        batch_size=args.batch_size,
                                        refidx=args.refidx,
                                        aug_level=args.aug_level
                                        )
    
    # model initialization
    model = MLP_SCA(target_dim=args.target_dim, 
                    feature_dim=args.feature_dim, 
                    class_dim=class_dim, 
                    g_hp=generator_hp, 
                    c_hp=classifier_hp, 
                    ref_noise=args.ref_noise
                    ).to(device)
    # print(f'Construct model complete, number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)# optimizer

    # log file
    log_file = get_filename('log', args.ref, args.target, args.target_byte, args.flag)
    file_handler = logging.FileHandler(log_file, mode='w')
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)

    print("################## Training Phase of LD-PA ##################")
    best_loss = float('inf') # for saving the best
    pre_loss = float('inf')
    counter, patience = 0, 3 # parameters for early stopping
    torch.cuda.empty_cache()
    for epoch in tqdm(range(0, args.epoch), desc='Training Epoch'):
        # train one epoch
        log_train = pairtrain_epoch(model, train_loader, valid_loader, optimizer, device, _alpha=args.alpha)

        log = {'epoch': epoch}
        log.update(log_train)
        if epoch==0: logger.info(', '.join(log.keys()))
        logger.info(', '.join(map(str, log.values())))

        # save the best model
        if log['valid_cls_loss'] < best_loss:
            best_loss = log['valid_cls_loss']
            counter = 0
            generator_hp, classifier_hp = model.get_hp()
            torch.save({
                'epoch': epoch,
                'g_hp': generator_hp,
                'c_hp': classifier_hp,
                'model_state_dict': model.state_dict()
                }, get_filename('model', args.ref, args.target, args.target_byte, args.flag))
            
        # activate early stopping
        if args.earlystop:
            if pre_loss > log['valid_cls_loss']:
                counter = 0
            elif epoch > 15: # start after epoch 15
                counter += 1
            if counter >= patience:
                print(f"Early stoping at epoch-{epoch}")
                break
            pre_loss = log['valid_cls_loss']
            
    logging.shutdown()


if __name__ == "__main__":
    run()
