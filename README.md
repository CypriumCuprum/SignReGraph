# SignReGraph
SignReGraph: Rethinking Skeleton-Based Sign Language Recognition with Masked Self-Supervision

## Installation
```bash
git clone https://github.com/CypriumCuprum/SignReGraph.git
cd SignReGraph
conda env create -f signregraph.yaml
conda activate signregraph
```

## Structure
```bash
SignReGraph/ 
├── configs/ 
├── signregraph/ 
├── tools/ 
├── data/ 
│   ├── signlanguage_asl_citizen/ 
│   │   └── 2731asl_citizen.pkl
│   ├── signlanguage_msasl/ 
│   │   ├── 1000msasl_focushand_w_face.pkl
│   │   ├── 200msasl_focushand_w_face.pkl
│   │   └── 100msasl_focushand_w_face.pkl
│   └── signlanguage_wlasl
│       ├── 2000wlasl_focus_hand_w_face.pkl
│       ├── 300wlasl_focus_hand_w_face.pkl
│       └── 100wlasl_focus_hand_w_face.pkl
├── workdirs/
│   ├── signlanguage_aslcitizen/
│   ├── signlanguage_msasl/
│   └── signlanguage_wlasl
├── requirements.txt
├── signregraph.yaml
├── README.md
└── .gitignore
```
## Data Preparation
Download the extracted skeleton data from [Skeleton Data](https://drive.google.com/drive/folders/1-JEbJlDZDSsaThkLHPew2-QqjiesnPX0?usp=drive_link) and place it into the directory as shown in the Structure.
## Training
*Please fill out "load_from" variable in WLASL and MSASL datasets. 
```bash
# Training
bash tools/dist_train.sh {config_path} {num_gpus} {other_options}

# Example:
bash tools/dist_train.sh configs/signlanguage_aslcitizen/j.py 1 --validate --test-last --test-best
```

Visualize recontructed frame
```bash
python -m tools.visualize_reconstructed_frame {config_path} -C {checkpoint} --input-file {Instance ID}
```

## Pretrained models and Our logging file
All the checkpoints can be downloaded from [here](https://drive.google.com/drive/folders/1heEZuEnTJyWXHc4pOItqr_wO5O4jW-WT?usp=drive_link).

## Acknowledgements
This repo is mainly based on [ProtoGCN](https://github.com/firework8/ProtoGCN.git) \
Thanks to the original authors for their excellent work!

## Contact
Email: riverflowsinyou1412@gmail.com
