
# Dataset description:

4D-DRESS contains 32 subjects and 64 real-world human outfits captured in 520 dynamic motion sequences for a total of 78k frames. One scan frame consists of one 80k-face triangle mesh with one texture uv map and vertex annotations, together with the multi-view RGB images with pixel labels and the registered SMPL(-X) body models. Totally, 4D-DRESS captures dynamic motions of 4 dresses, 30 upper, 28 lower, and 32 outer garments. For each garment, we also provide its canonical template mesh to benefit the clothing simulation study.

|Subj/ID|  Gender  |  Outfit |#Seqs| #Frames |  Outfit |#Seqs| #Frames |
| :---: |  :----:  |  :---:  | :-: | :-----: |  :---:  | :-: | :-----: |
| 00122 |   Male   |  Inner  |  5  |   730   |  Outer  |  6  |   950   |
| 00123 |  Female  |  Inner  |  6  |   920   |  Outer  |  6  |   950   |
| 00127 |   Male   |  Inner  |  8  |  1175   |  Outer  |  8  |  1230   |
| 00129 |  Female  |  Inner  |  8  |  1170   |  Outer  |  8  |  1175   |
| 00134 |   Male   |  Inner  |  8  |  1100   |  Outer  |  7  |  1015   |
| 00135 |   Male   |  Inner  |  8  |  1250   |  Outer  |  9  |  1470   |
| 00136 |  Female  |  Inner  |  9  |  1290   |  Outer  |  10 |  1540   |
| 00137 |  Female  |  Inner  |  10 |  1600   |  Outer  |  10 |  1450   |
| 00140 |  Female  |  Inner  |  9  |  1350   |  Outer  |  10 |  1465   |
| 00147 |  Female  |  Inner  |  9  |  1280   |  Outer  |  9  |  1310   |
| 00148 |  Female  |  Inner  |  9  |  1350   |  Outer  |  6  |   910   |
| 00149 |   Male   |  Inner  |  9  |  1330   |  Outer  |  9  |  1390   |
| 00151 |  Female  |  Inner  |  7  |  1080   |  Outer  |  8  |  1320   |
| 00152 |  Female  |  Inner  |  8  |  1210   |  Outer  |  8  |  1305   |
| 00154 |   Male   |  Inner  |  9  |  1330   |  Outer  |  10 |  1610   |
| 00156 |  Female  |  Inner  |  8  |  1140   |  Outer  |  8  |  1320   |
| 00160 |   Male   |  Inner  |  8  |  1310   |  Outer  |  8  |  1230   |
| 00163 |  Female  |  Inner  |  9  |  1380   |  Outer  |  8  |  1180   |
| 00167 |  Female  |  Inner  |  7  |  1070   |  Outer  |  8  |  1070   |
| 00168 |   Male   |  Inner  |  7  |  1050   |  Outer  |  8  |  1340   |
| 00169 |   Male   |  Inner  |  8  |  1130   |  Outer  |  9  |  1290   |
| 00170 |  Female  |  Inner  |  9  |  1360   |  Outer  |  8  |  1200   |
| 00174 |   Male   |  Inner  |  9  |  1260   |  Outer  |  8  |  1150   |
| 00175 |   Male   |  Inner  |  8  |  1340   |  Outer  |  10 |  1620   |
| 00176 |  Female  |  Inner  |  7  |  1090   |  Outer  |  8  |  1240   |
| 00179 |   Male   |  Inner  |  8  |  1250   |  Outer  |  7  |  1090   |
| 00180 |   Male   |  Inner  |  8  |  1200   |  Outer  |  8  |  1220   |
| 00185 |  Female  |  Inner  |  9  |  1320   |  Outer  |  9  |  1320   |
| 00187 |  Female  |  Inner  |  9  |  1280   |  Outer  |  9  |  1260   |
| 00188 |   Male   |  Inner  |  8  |  1215   |  Outer  |  8  |  1220   |
| 00190 |  Female  |  Inner  |  7  |  1070   |  Outer  |  7  |  1010   |
| 00191 |  Female  |  Inner  |  7  |  1040   |  Outer  |  8  |  1270   |


# Dataset structure:
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


# Useful tools for 4D-DRESS:
Visualize 4D-DRESS sequences using [aitviewer](https://github.com/eth-ait/aitviewer).
```
python visualize.py --subj 00122  --outfit Outer --seq Take9
```

Extract labeled cloth meshes and render multi-view pixel labels using vertex annotations.
```
python extract_garment.py --subj 00122  --outfit Outer --seq Take9
```

# Benchmark sequences:
The sequences used for benchmark evaluations can be found in `benchmarks.py`.

- Template-based clothing simulation is conducted on 16 garments and 96 sequences.

| Outfit | Cloth | Subj/ID |                       Sequences                       |
| :----: | :---: |  :---:  |  :-------------------------------------------------:  |
| Inner  | lower |  00129  | 'Take2', 'Take3', 'Take4', 'Take5', 'Take6', 'Take22' |
| Inner  | lower |  00152  | 'Take2', 'Take3', 'Take4', 'Take5', 'Take6', 'Take7'  |
| Inner  | lower |  00156  | 'Take2', 'Take3', 'Take4', 'Take7', 'Take8', 'Take9'  |
| Inner  | lower |  00174  | 'Take2', 'Take3', 'Take4', 'Take5', 'Take6', 'Take7'  |
| Inner  | upper |  00127  | 'Take5', 'Take6', 'Take7', 'Take8', 'Take9', 'Take10' |
| Inner  | upper |  00140  | 'Take1', 'Take3', 'Take4', 'Take6', 'Take7', 'Take8'  |
| Inner  | upper |  00147  | 'Take1', 'Take2', 'Take3', 'Take4', 'Take6', 'Take9'  |
| Inner  | upper |  00180  | 'Take2', 'Take3', 'Take4', 'Take5', 'Take6', 'Take7'  |
| Inner  | dress |  00148  | 'Take4', 'Take5', 'Take6', 'Take7', 'Take8', 'Take9'  |
| Inner  | dress |  00170  | 'Take1', 'Take3', 'Take5', 'Take7', 'Take8', 'Take9'  |
| Inner  | dress |  00185  | 'Take1', 'Take2', 'Take3', 'Take4', 'Take7', 'Take8'  |
| Inner  | dress |  00187  | 'Take1', 'Take2', 'Take3', 'Take4', 'Take5', 'Take6'  |
| Outer  | outer |  00123  | 'Take8', 'Take9', 'Take10', 'Take11', 'Take12', 'Take13' |
| Outer  | outer |  00152  | 'Take10', 'Take12', 'Take15', 'Take17', 'Take18', 'Take19'  |
| Outer  | outer |  00176  | 'Take9', 'Take10', 'Take11', 'Take12', 'Take13', 'Take14'  |
| Outer  | outer |  00190  | 'Take10', 'Take11', 'Take13', 'Take14', 'Take15', 'Take16'  |

- Single-view human&cloth reconstruction and image-based human parsing are conducted on 64 outfits and 128 sequences.

|Subj/ID| Outfit |    Sequences   | Outfit |    Sequences   |
| :---: | :----: | :------------: | :----: | :------------: |
| 00122 | Inner  |'Take5', 'Take8'| Outer  |'Take11', 'Take16'|
| 00123 | Inner  |'Take3', 'Take5'| Outer  |'Take10', 'Take11'|
| 00127 | Inner  |'Take8', 'Take9'| Outer  |'Take16', 'Take18'|
| 00129 | Inner  |'Take3', 'Take5'| Outer  |'Take11', 'Take13'|
| 00134 | Inner  |'Take5', 'Take6'| Outer  |'Take12', 'Take19'|
| 00135 | Inner  |'Take7', 'Take10'| Outer  |'Take21', 'Take24'|
| 00136 | Inner  |'Take8', 'Take12'| Outer  |'Take19', 'Take28'|
| 00137 | Inner  |'Take5', 'Take7'| Outer  |'Take16', 'Take19'|
| 00140 | Inner  |'Take6', 'Take8'| Outer  |'Take19', 'Take21'|
| 00147 | Inner  |'Take11', 'Take12'| Outer  |'Take16', 'Take19'|
| 00148 | Inner  |'Take6', 'Take7'| Outer  |'Take16', 'Take19'|
| 00149 | Inner  |'Take4', 'Take12'| Outer  |'Take14', 'Take24'|
| 00151 | Inner  |'Take4', 'Take9'| Outer  |'Take15', 'Take20'|
| 00152 | Inner  |'Take4', 'Take8'| Outer  |'Take17', 'Take18'|
| 00154 | Inner  |'Take5', 'Take9'| Outer  |'Take20', 'Take21'|
| 00156 | Inner  |'Take4', 'Take8'| Outer  |'Take14', 'Take19'|
| 00160 | Inner  |'Take6', 'Take7'| Outer  |'Take17', 'Take18'|
| 00163 | Inner  |'Take7', 'Take10'| Outer  |'Take13', 'Take15'|
| 00167 | Inner  |'Take7', 'Take9'| Outer  |'Take12', 'Take14'|
| 00168 | Inner  |'Take3', 'Take7'| Outer  |'Take11', 'Take16'|
| 00169 | Inner  |'Take3', 'Take10'| Outer  |'Take17', 'Take19'|
| 00170 | Inner  |'Take9', 'Take11'| Outer  |'Take15', 'Take24'|
| 00174 | Inner  |'Take6', 'Take9'| Outer  |'Take13', 'Take15'|
| 00175 | Inner  |'Take4', 'Take9'| Outer  |'Take13', 'Take20'|
| 00176 | Inner  |'Take3', 'Take6'| Outer  |'Take11', 'Take14'|
| 00179 | Inner  |'Take4', 'Take8'| Outer  |'Take13', 'Take15'|
| 00180 | Inner  |'Take3', 'Take7'| Outer  |'Take14', 'Take17'|
| 00185 | Inner  |'Take7', 'Take8'| Outer  |'Take17', 'Take18'|
| 00187 | Inner  |'Take4', 'Take6'| Outer  |'Take10', 'Take15'|
| 00188 | Inner  |'Take7', 'Take8'| Outer  |'Take12', 'Take18'|
| 00190 | Inner  |'Take2', 'Take7'| Outer  |'Take14', 'Take17'|
| 00191 | Inner  |'Take3', 'Take6'| Outer  |'Take13', 'Take19'|

- Video-based human reconstruction and human representation learning are conducted on 8 outfits.

| Outfit | Subj/ID|                Train Sequences               |   Test   |
| :----: | :----: | :------------------------------------------: |  :----:  |
| Inner  | 00148  |'Take1', 'Take2', 'Take4', 'Take5', 'Take6', 'Take8', 'Take9', 'Take10'|'Take7'|
| Inner  | 00152  |'Take2', 'Take3', 'Take4', 'Take5', 'Take6', 'Take7', 'Take9'|'Take8'|
| Inner  | 00154  |'Take1', 'Take3', 'Take4', 'Take5', 'Take6', 'Take7', 'Take8', 'Take11'|'Take9'|
| Inner  | 00185  |'Take1', 'Take2', 'Take3', 'Take4', 'Take5', 'Take6', 'Take8', 'Take9'|'Take7'|
| Outer  | 00127  |'Take11', 'Take13', 'Take14', 'Take15', 'Take16', 'Take17', 'Take19'|'Take18'|
| Outer  | 00137  |'Take12', 'Take13', 'Take14', 'Take15', 'Take17', 'Take18', 'Take19', 'Take20', 'Take21'|'Take16'|
| Outer  | 00149  |'Take14', 'Take15', 'Take16', 'Take17', 'Take20', 'Take22', 'Take24', 'Take25'|'Take21'|
| Outer  | 00188  |'Take10', 'Take11', 'Take12', 'Take15', 'Take16', 'Take17', 'Take18'|'Take14'|





