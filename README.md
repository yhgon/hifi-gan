# HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis

modified by hyungon ryu 

## Pre-requisites
1. Python >= 3.6
2. Clone this repository.

    -- modify to configure checkpoint/config files `inference.py`
    -- add tic/toc for inference `inference.py`   
    -- add warm up to measure accurate tic/toc `inference.py`
```bash
$git clone https://github.com/yhgon/hifi-gan.git
```

3. Install python requirements. Please refer [requirements.txt](requirements.txt)
4. Download and extract the [LJ Speech dataset](https://keithito.com/LJ-Speech-Dataset/).
And move all wav files to `LJSpeech-1.1/wavs`


## Pretrained Model
download  pretrained models for LJSpeech which authors provided from Google drive.<br/>
```bash
$wget https://raw.githubusercontent.com/yhgon/colab_utils/master/gfile.py
$mkdir checkpoints
$python gfile.py -u 'https://drive.google.com/file/d/14NENd4equCBLyyCSke114Mv6YR_j_uFs/view?usp=sharing'  -d 'checkpoints' -f 'LJ_v1.pt'  # LJ_V1
$python gfile.py -u 'https://drive.google.com/file/d/1gfouaWecMbmfqIdWYs-KtsULIdYCveYW/view?usp=sharing'  -d 'checkpoints' -f 'LJ_v2.pt'  # LJ_V2
$python gfile.py -u 'https://drive.google.com/file/d/18TNnHbr4IlduAWdLrKcZrqmbfPOed1pS/view?usp=sharing'  -d 'checkpoints' -f 'LJ_v3.pt'  # LJ_V3
```


## Inference from wav file
1. Make `input` directory and copy wav files into the directory.
2. Run the following command.
```
$python inference.py --checkpoint_file checkpoints/LJ_v1.pt --config_file config_v1.json  --input_wavs_dir input --output_dir out_v1
```
Generated wav files are saved in `out_v1` .<br>
 


## Inference for end-to-end speech with FastPitch
`TODO` 

## preprocess 
`TODO`

## Training
```
python train.py --config config_v1.json
```
To train V2 or V3 Generator, replace `config_v1.json` with `config_v2.json` or `config_v3.json`.<br>
Checkpoints and copy of the configuration file are saved in `cp_hifigan` directory by default.<br>
You can change the path by adding `--checkpoint_path` option.





## Acknowledgements
We referred to [WaveGlow](https://github.com/NVIDIA/waveglow), [MelGAN](https://github.com/descriptinc/melgan-neurips) 
and [Tacotron2](https://github.com/NVIDIA/tacotron2) to implement this.

