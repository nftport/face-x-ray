# Face-X-ray

This repo contains code for [Face X-ray for More General Face Forgery Detection](https://arxiv.org/abs/1912.13458).

Modifications -

The authors have not explicitly defined the image segmentation network, here we have used DeepLabV3Plus with efficientnet-b6 backbone.
The model achieves excellent accuracy for faceswapped deepfakes. 

## Usage

### Dataset generation

To build the dataset of blended images

1. Download dlib shape_predictor_68_face_landmarks

```
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
```

2. Extract bz2 file to get .dat file

3. To generate the landmarks, add image file folder in landmark_generation.py and then run:

```
python landmark_generation.py
```

4. Finally to generate the blended images:

```
python bi_online_generation.py
```

### Training

To train:

Point the path to the folder of blended images

```
python train_model.py
```

For distributed training
```
python train_model_distributed.py
```

