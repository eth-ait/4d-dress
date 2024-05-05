# <p align="center"> 4D-DRESS: A 4D Dataset of Real-world Human Clothing with Semantic Annotations </p>

###  <p align="center"> [Wenbo Wang*](https://wenbwa.github.io), [Hsuan-I Ho*](https://ait.ethz.ch/people/hohs), [Chen Guo](https://ait.ethz.ch/people/cheguo), [Boxiang Rong](https://ribosome-rbx.github.io), [Artur Grigorev](https://ait.ethz.ch/people/agrigorev), [Jie Song](https://ait.ethz.ch/people/song), [Juan Jose Zarate](https://ait.ethz.ch/people/jzarate), [Otmar Hilliges](https://ait.ethz.ch/people/hilliges) </p>

### <p align="center"> [CVPR 2024 Highlight](https://cvpr2023.thecvf.com) </p>

## <p align="center"> [ArXiv](https://arxiv.org/abs/2404.18630) / [Video](https://www.youtube.com/watch?v=dEQ4dvO8BsE) / [Dataset](https://4d-dress.ait.ethz.ch) / [Website](https://eth-ait.github.io/4d-dress/) </p>

<p align="center">
  <img width="100%" src="assets/4D-DRESS.png"/>
</p>
  <strong>Dataset Overview.</strong> 4D-DRESS is the first real-world 4D dataset of human clothing, capturing <strong>64</strong> human outfits in more than <strong>520</strong> motion sequences and <strong>78k</strong> scan frames. Each motion sequence includes a) high-quality 4D textured scans; for each textured scan, we annotate b) precise vertex-level semantic labels, thereby obtaining c) the corresponding extracted garment meshes and fitted SMPL(-X) body meshes. Totally, 4D-DRESS captures dynamic motions of 4 dresses, 28 lower, 30 upper, and 32 outer garments. For each garment, we also provide its canonical template mesh to benefit the future human clothing study.
</p>
<hr>

## Released
- [x] 4D-DRESS Dataset.
- [x] 4D-Human-Parsing Code.

## Env Installation
Git clone this repo:
```
git clone -b main --single-branch https://github.com/eth-ait/4d-dress.git
cd 4d-dress
```

Create conda environment from environment.yaml:
```
conda env create -f environment.yml
conda activate 4ddress
```
Or create a conda environment via the following commands:
```
conda create -n 4ddress python==3.8
conda activate 4ddress
bash env_install.sh
```

## Model Installation
Install image-based parser: [Graphonomy](https://github.com/Gaoyiminggithub/Graphonomy.git).

Download checkpoint inference.pth from [here](https://drive.google.com/file/d/1eUe18HoH05p0yFUd_sN6GXdTj82aW0m9/view?usp=sharing) and save to 4dhumanparsing/checkpoints/graphonomy/.
```
git clone https://github.com/Gaoyiminggithub/Graphonomy.git 4dhumanparsing/lib/Graphonomy
mkdir 4dhumanparsing/checkpoints
mkdir 4dhumanparsing/checkpoints/graphonomy
```

Install optical flow predictor: [RAFT](https://github.com/princeton-vl/RAFT).

Download checkpoint raft-things.pth and save to 4dhumanparsing/checkpoints/raft/models/.
```
git clone https://github.com/princeton-vl/RAFT.git 4dhumanparsing/lib/RAFT
wget -P 4dhumanparsing/checkpoints/raft/ https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip
unzip 4dhumanparsing/checkpoints/raft/models.zip -d 4dhumanparsing/checkpoints/raft/
```

Install segment anything model: [SAM](https://github.com/facebookresearch/segment-anything).

Download checkpoint sam_vit_h_4b8939.pth and save to 4dhumanparsing/checkpoints/sam/.
```
pip install git+https://github.com/facebookresearch/segment-anything.git
wget -P 4dhumanparsing/checkpoints/sam/ https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Install and compile graph-cut optimizer: [pygco](https://github.com/yujiali/pygco.git):
```
git clone https://github.com/yujiali/pygco.git 4dhumanparsing/lib/pygco
cd 4dhumanparsing/lib/pygco
wget -N -O gco-v3.0.zip https://vision.cs.uwaterloo.ca/files/gco-v3.0.zip
unzip -o gco-v3.0.zip -d  ./gco_source
make all
cd ../../..
```

## Download Dataset
Please download the 4D-DRESS dataset and place the folders according to the following structure:

    4D-DRESS
    └── < Subject ID > (00***)
        └── < Outfit > (Inner, Outer)
           └── < Sequence ID > (Take*)
                ├── basic_info.pkl: {'scan_frames', 'rotation', 'offset', ...}
                ├── Meshes_pkl
                │   ├── atlas-fxxxxx.pkl: uv texture map as pickle file (1024, 1024, 3)
                │   └── mesh-fxxxxx.pkl: {'vertices', 'faces', 'colors', 'normals', 'uvs'}
                ├── SMPL
                │   ├── mesh-fxxxxx_smpl.pkl: SMPL params
                │   └── mesh-fxxxxx_smpl.ply: SMPL mesh
                ├── SMPLX
                │   ├── mesh-fxxxxx_smplx.pkl: SMPLX params
                │   └── mesh-fxxxxx_smplx.ply: SMPLX mesh
                ├── Semantic
                │   ├── labels
                │   │   └── label-fxxxxx.pkl, {'scan_labels': (nvt, )}
                │   ├── clothes: let user extract
                │   │   └── cloth-fxxxxx.pkl, {'upper': {'vertices', 'faces', 'colors', 'uvs', 'uv_path'}, ...}
                ├── Capture
                │   ├── cameras.pkl: {'cam_id': {"intrinsics", "extrinsics", ...}}
                │   ├── < Camera ID > (0004, 0028, 0052, 0076)
                │   │   ├── images
                │   │   │   └── capture-f*****.png: captured image (1280, 940, 3) 
                │   │   ├── masks
                │   │   │   └── mask-f*****.png: rendered mask (1280, 940)
                │   │   ├── labels: let user extract
                │   │   │   └── label-f*****.png: rendered label (1280, 940, 3) 
                └── └── └── └── overlap-f*****.png: overlapped label (1280, 940, 3)


## Useful tools for 4D-DRESS:
Visualize 4D-DRESS sequences using [aitviewer](https://github.com/eth-ait/aitviewer).
```
python dataset/visualize.py --subj 00122  --outfit Outer --seq Take9
```

Extract labeled cloth meshes and render multi-view pixel labels using vertex annotations.
```
python dataset/extract_garment.py --subj 00122  --outfit Outer --seq Take9
```

<hr>
<p align="center">
  <img width="100%" src="assets/4DHumanParsing.png"/>
</p>
  <strong>4D Human Parsing Method.</strong> We first render current and previous frame scans into multi-view images and labels. 3.1) Then collect multi-view parsing results from the image parser, optical flows, and segmentation masks. 3.2) Finally, we project multi-view labels to 3D vertices and optimize vertex labels using the Graph Cut algorithm with vertex-wise unary energy and edge-wise binary energy. 3.3) The manual rectification labels can be easily introduced by checking the multi-view rendered labels.
</p>
<hr>

## 4D Human Parsing on 4D-DRESS
First, run the image parser, optical flow, and Segment Anything models on the entire 4D scan sequence, and parse the first frame:
```
python 4dhumanparsing/multi_view_parsing.py --subj 00122  --outfit Outer --seq Take9
```
Second, run graph-cut optimization and introduce manual rectification on the entire 4D scan sequence:
```
python 4dhumanparsing/multi_surface_parsing.py --subj 00122  --outfit Outer --seq Take9
``` 

## 4D Human Parsing with New Labels
You can introduce new labels, like socks and belts, during the 4D human parsing process.

First, run the image parser, optical flow, and Segment Anything models, parse the first frame with new_label=sock:
```
python 4dhumanparsing/multi_view_parsing.py --subj 00135  --outfit Inner --seq Take1 --new_label sock
```
Second, run graph-cut optimization and introduce manual rectification on all frames with new_label=sock:
```
python 4dhumanparsing/multi_surface_parsing.py --subj 00135  --outfit Inner --seq Take1 --new_label sock
```
Tracking and parsing small regions like socks and belts may need more manual rectification efforts.

## 4D Human Parsing on Other Datasets
You can apply our 4D human parsing method on other 4D human datasets, like [BUFF](https://buff.is.tue.mpg.de/), [X-Humans](https://github.com/Skype-line/X-Avatar), and [Actors-HQ](https://www.actors-hq.com/).

For instance, you can modify our DatasetUtils within 4dhumanparsing/lib/utility/dataset.py to XHumansUtils.

And then, like before, run image parser, optical flow, and segment anything models on X-Humans sequence:
```
python 4dhumanparsing/multi_view_parsing.py --dataset XHumans --subj 00017  --outfit test --seq Take10
```
After which, run graph-cut optimization and introduce manual rectification on all frames:
```
python 4dhumanparsing/multi_surface_parsing.py --dataset XHumans --subj 00017  --outfit test --seq Take10
```



## Related Work

* Yin et. al, "[Hi4D: 4D Instance Segmentation of Close Human Interaction](https://yifeiyin04.github.io/Hi4D/)", CVPR 2023
* Shen et. al, "[X-Avatar: Expressive Human Avatars](https://skype-line.github.io/projects/X-Avatar/)", CVPR 2023
* Antić et. al, "[CloSe: A 3D Clothing Segmentation Dataset and Model](https://virtualhumans.mpi-inf.mpg.de/close3dv24/)", 3DV 2024

If you find our code, dataset, and paper useful, please cite as
```
@inproceedings{wang20244ddress,
title={4D-DRESS: A 4D Dataset of Real-world Human Clothing with Semantic Annotations},
author={Wang, Wenbo and Ho, Hsuan-I and Guo, Chen and Rong, Boxiang and Grigorev, Artur and Song, Jie and Zarate, Juan Jose and Hilliges, Otmar},
booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2024}
}
```
