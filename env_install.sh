echo "Installing torch..."
echo "If necessary, change to other versions (default python3.8 cuda11.3)"
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
conda install pytorch-scatter -c pyg

echo "Installing pytorch3d..."
echo "Following https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md"
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html

echo "Installing requirements..."
pip install -r requirements.txt
