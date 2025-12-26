pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.19.0 
pip install mmsegmentation==0.19.0 
pip install timm

cd ./mmdetection3d
pip install -e .

cd ../projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn
python setup.py build install

cd /tonyxu
pip install -r requirement.txt

mkdir ckpts
cd ckpts
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
wget https://download.pytorch.org/models/resnet18-f37072fd.pth