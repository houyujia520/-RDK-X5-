先安装Python 3.8+


下载miniconda: https://www.anaconda.com/download


打开anaconda的prompt，进入项目目录


创建Conda环境
conda create -n rppg_realtime python=3.8 -y


激活环境
conda activate rppg_realtime


安装PyTorch (CUDA 11.8版本)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


安装其他依赖
pip install -r requirements.txt


运行: python main.py