## Conda environment

```bash
conda create -n crowd python=3.10
conda activate crowd
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install pytorch-lightning wandb hydra-core hydra-colorlog
```
