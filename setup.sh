python -m pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install kaolin==0.16.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.3.0_cu121.html 
python -m pip install torch-cluster -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
python -m pip install -r requirements.txt
python -m pip install gs/submodules/simple-knn
python -m pip install gs/submodules/diff-gaussian-rasterization
python -m pip install git+https://github.com/jukgei/diff-gaussian-rasterization.git@b1e1cb83e27923579983a9ed19640c6031112b94
python -m pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8