# FreqFusion + Faster R-CNN (MMDetection 2.x)

Repositorio para comparar Faster R-CNN (baseline) contra Faster R-CNN con FreqFusion usando MMDetection 2.x. Está pensado para correr rápido y sin rodeos, con scripts y configs simples.

## Requisitos
Python 3.9–3.10, PyTorch con CUDA compatible con tu GPU, mmcv-full 1.5.3 y mmdet 2.28.1. Si ya tienes un entorno funcional, pasa directo a “Entrenamiento”.

## Instalación (Conda)
conda create -n freqfusion python=3.10 -y
conda activate freqfusion

# PyTorch: elige la rueda de tu versión de CUDA (ejemplo CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# MMDetection 2.x
pip install mmcv-full==1.5.3
pip install mmdet==2.28.1

# Utilidades
pip install opencv-python tqdm matplotlib

