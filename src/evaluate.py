import torch
import os
import numpy as np
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from dataloader import load_file
from utils import plot_GEnSR, plot_snr, compute_label_key_guess, compute_iv
from preprocess import standardization
from network import MLP_SCA
from sca_metrics import sca_metrics, SNR_fit

no_cuda =False
cuda = not no_cuda and torch.cuda.is_available()
device = torch.device('cuda') if cuda else torch.device('cpu')


def run():
    parser = argparse.ArgumentParser(description="Arguments for model evaluation.")
    parser.add_argument("-fp", "--file_path", type=str, default='./data', help="Path to data")
    parser.add_argument("-mp", "--model_path", type=str, default=None, help="Path to the model checkpoint")
    parser.add_argument("--target", type=str, default='ASCADf', help="Target dataset")
    parser.add_argument("--target_byte", type=int, default=0, choices=np.arange(0,16,1), help="Targeting byte")
    parser.add_argument("--feature_dim", type=int, default=100, help="Dimension of reference features.")
    parser.add_argument("--target_dim", type=int, default=10000, help="Dimension of target traces.")
    parser.add_argument("--window", type=int, default=20, help="Downsampling window.")
    parser.add_argument("-lm", "--labeling_method", type=str, default='ID', choices=['ID', 'HW'], help="Labeling method")
    parser.add_argument("-ntv", "--nb_tar_valid", type=int, default=5000, help="Number of target traces for validation")
    parser.add_argument("-nta", "--nb_tar_attack", type=int, default=5000, help="Number of target traces for attacks")

    args = parser.parse_args()

    tar_dataset = f'{args.file_path}/{args.target}_nopoi_{args.window}window.h5'
    class_dim = 256 if args.labeling_method=='ID' else 9

    ## load model from checkpoint
    checkpoint = torch.load(args.model_path)
    print(f"Model at epoch {checkpoint['epoch']} loaded")
    model = MLP_SCA(target_dim=args.target_dim, 
                    feature_dim=args.feature_dim, 
                    class_dim=class_dim, 
                    g_hp=checkpoint['g_hp'], 
                    c_hp=checkpoint['c_hp']
                    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # load attack traces and metadata for evaluation
    tar_traces_attack, tar_plaintext_attack, tar_key_attack, _ = load_file(tar_dataset, args.nb_tar_valid, args.nb_tar_attack, profile=False)
    tar_plaintext_attack = tar_plaintext_attack[:, args.target_byte]
    tar_key_attack = tar_key_attack[:, args.target_byte]
    tar_traces_attack = standardization(tar_traces_attack, desc='Standardlizing target attacking traces')

    # model prediction
    with torch.no_grad():
        feature, prediction = model(torch.tensor(tar_traces_attack).float().to(device))

    print("Compute and plot SNR for sbox's output.")
    feature = feature.cpu().numpy()
    nb_cls = 256 if args.labeling_method=='ID' else 9
    label_attack = compute_iv(tar_plaintext_attack, tar_key_attack, None, args.labeling_method)
    snr_val_sbox = SNR_fit(feature, nb_cls, label_attack)
    plot_snr(snr_val_sbox, savefile='snr_val_sbox.png', flag=["SNR for sbox's output"], show=False)
    print("Figure of SNR for sbox's output is saved as snr_val_sbox.png")

    print("Compute and plot guessing entropy")
    label_key_guess = compute_label_key_guess(tar_plaintext_attack, args.labeling_method)
    prediction = torch.nn.functional.softmax(prediction, -1)
    prediction = torch.log(prediction + 1e-36)
    guessing_entropy, _, _ = sca_metrics(prediction.cpu().numpy(), args.nb_tar_attack, label_key_guess, tar_key_attack[0])
    plot_GEnSR(guessing_entropy, None, savefile='GE.png', show=False)
    print("Figure of guessing entropy is saved as GE.png")
        


if __name__ == "__main__":
    run()
