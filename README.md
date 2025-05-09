# CMD
Code for paper A Cell-interacting and Multi-correcting Method for Automatic Circulating Tumor Cells Detection. 

## Method
<p align="center"><img width="800" src="https://github.com/zx333445/CMD/blob/main/flow.png?raw=true"></p>

## Usage

Directory description:

```
├─ netcmd              // directory of designed method CMD
├─ network             // directory of detection networks
├─ tool                // directory of tool codes

├─ datasets.py         // dataset code
├─ train.py            // main code for model training
├─ trainer.py          // code for training utils
├─ launch.sh           // train.py launcher
├─ _utils.py           // other utils code
```

Run the following code for model training:

```bash
$ bash launch.sh
```

This will initiate the script `train.py` for 5-fold cross-validation model training.

## Infer
This file gives the code and examples of model inference [infer.ipynb](./infer.ipynb)

You can download the trained model parameters at this link [CMD Model Weights](https://drive.google.com/file/d/1FKQzjePb4N0TolI7seGhXPOZVtQ8OKH5/view?usp=drive_link)
