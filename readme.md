# This is the source code for the paper "LD-PA: Distilling Univariate Leakage for Deep Learning-based Profiling Attacks".

## LD-PA framework:
![](framework.bmp)

## Usage:
1. Preparing the target dataset and downsampling (if needed).
2. <a href=./src/train.py>Train</a> a model with LD-PA paradigm. 
3. <a href=./src/evaluate.py>Evaluate</a> the model. 

## An example of LD-PA is given in the Jupyter Notebook <a href=example.ipynb>file</a>.
- Training loss <a href=log_ASCADv_ASCADf2_s0.log>log</a>
- Saved best model at epoch 17 (./model_ASCADv_ASCADf2_s0.pth).
- Evaluation results: 
    - SNR for sbox's output.
        ![](snr_val_sbox.png)
    - Guessing Entropy.
        ![](GE.png)


## Citation:
If you find this work useful, please consider citing:
```
@artical{Xiao2024ldpa,
  title={LD-PA: Distilling Univariate Leakage for Deep Learning-based Profiling Attacks},
  author={Chong Xiao and Ming Tang and Sengim Karayalcin and Wei Cheng},
  journal={IEEE Transactions on Information Forensics and Security},
  volume={},
  number={},
  pages={--},
  year={2024}
}
```