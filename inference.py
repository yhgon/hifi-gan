from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator

import time 

h = None
device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)


def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    tic_start = time.time()

    filelist = os.listdir(a.input_wavs_dir)
    os.makedirs(a.output_dir, exist_ok=True)

    toc_filelist = time.time()

    generator = Generator(h).to(device)
    toc_model = time.time() 


    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])
    toc_checkpoint = time.time() 



    generator.eval()
    generator.remove_weight_norm()
    toc_ready = time.time() 

    with torch.no_grad():
        for i, filname in enumerate(filelist):
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            tic_gen = time.time()
            x = get_mel(wav.unsqueeze(0))
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            toc_gen = time.time()
            dur_gen = toc_gen - tic_gen
            audio = audio.cpu().numpy().astype('int16')
            dur_wav = len(audio)
            sec_wav = dur_wav / h.sampling_rate

            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated.wav')
            write(output_file, h.sampling_rate, audio)
            print("it took  {:06.4f}sec for {:7.2f}sec  {:d}sample  audio {:4.1f} Msamples/sec RTF {:5.4f} {}   ".format( dur_gen, sec_wav ,dur_wav,   (h.sampling_rate*sec_wav/dur_gen)/(1000000), dur_gen/sec_wav,  output_file)   )


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    parser.add_argument('--config_file', required=True)    
    a = parser.parse_args()

    config_file = a.config_file
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)
    print(h)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()

