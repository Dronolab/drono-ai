# drono-ai

# predict.py

main script used to launch prediction and trainings using the available models from ./models and YOLO datasets from /data.

## Training

An NVIDIA graphic card is strongly recommended to attempt training. 
Two options are available: 
1. Use your own graphic card 
2. Use the google collab

Steps for using own graphic card:
1. check your CUDA version with ```nvidia-smi``` the graphic cards need to be recent enough and have enough VRAM.
2. download the appropriate version of torch, torchvision and torchaudio
3. add a dataset in the data folder *it must be the appropriate format ex: YOLOv11*
4. make sure you have .pt files in the folder models
5. run the predict.py script and follow the indications

Steps for using google collab:

*Available but instructions not written yet*


# open_cv_circle_annotation.py

small pre-annotation scrip used to annotate the easier images. it can and should be optimized more as it's current state is average. 
