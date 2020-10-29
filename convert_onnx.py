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

    generator = Generator(h).to(device).half()
    toc_model = time.time() 


    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])
    toc_checkpoint = time.time() 


    generator.half() ############ half
    generator.eval()
    generator.remove_weight_norm()
    toc_ready = time.time() 


    


    with torch.no_grad():
        ## try onnx 
        print("try onnx ")
        mel_rand = torch.randn(1, 80, 620).cuda().half() 

        print("try onnx export")
        torch.onnx.export(generator, mel_rand, a.onnx_filename, 
            input_names=["mel"], 
            output_names=["wav"], 
            dynamic_axes={ "mel":  {0: "batch_size", 2: "mel_seq"},
                           "wav":  {0: "batch_size", 1: "wav_seq"}})
        print("done")

        ## warm up 
        print("warm up..." , end='')
        for i, filname in enumerate(filelist):
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device) 
            wav.half() ############ half
            x = get_mel(wav.unsqueeze(0))
            xx=x.half()
            ############ half
            y_g_hat = generator( xx)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            toc_gen = time.time()          
        print("done")
        for i, filname in enumerate(filelist):
            tic_loadwav = time.time()
            wav, sr = load_wav(os.path.join(a.input_wavs_dir, filname))
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            wav.half() ############ half
            toc_loadwav = time.time()
            x = get_mel(wav.unsqueeze(0))
            xx=x.half()
            toc_mel = time.time()
            y_g_hat = generator(xx ) 
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            toc_gen = time.time()
            
            audio = audio.cpu().numpy().astype('int16')
            toc_wav_cpu = time.time()
            dur_wav = len(audio)                   
            sec_wav = dur_wav / h.sampling_rate
            
            dur_loadwav = toc_loadwav - tic_loadwav
            dur_mel     = toc_mel     - toc_loadwav
            dur_gen     = toc_gen     - toc_mel
            dur_wav_cpu = toc_wav_cpu - toc_gen
            dur_all     = toc_wav_cpu - toc_mel

            output_file = os.path.join(a.output_dir, os.path.splitext(filname)[0] + '_generated.wav')
            write(output_file, h.sampling_rate, audio)
            print("it took  {:06.4f}sec for {:7.2f}sec  {:d}sample  audio {:4.1f} Msamples/sec RTF {:5.4f} {}   ".format( dur_gen, sec_wav ,dur_wav,   (h.sampling_rate*sec_wav/dur_gen)/(1000000), dur_gen/sec_wav,  output_file)   )


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--onnx_filename', default='./generator.onnx')
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
    #print(h)

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

