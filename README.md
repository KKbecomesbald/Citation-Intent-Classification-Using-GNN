# Citation-Intent-Classification-Using-GNN

### Python environment setup with Conda

Tested with Python 3.9/3.10, PyTorch 2.2.0, and PyTorch Geometric 2.3.1.

To set up the environment, run the following commands:
```bash
conda create -n GNN python=3.10
conda activate GNN

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric==2.3.1
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.2.0+cu118.html

pip install scikit-learn==1.4.0
pip install fsspec rdkit
pip install pytorch-lightning yacs torchmetrics
pip install networkx
pip install tensorboardX
pip install ogb
pip install wandb
```


### Running Training

To execute training, use the following format for executing training runs:
```bash
conda activate GNN
python main.py --cfg configs/rgcn/acl-arc.yaml --repeat 2 seed 0
```

