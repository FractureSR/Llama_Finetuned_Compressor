# this script prepares data for audio data sft
# at the same time text compression ratio using flac


from glob import glob
audios = []

dataset = 'LibriSpeech' # LibriSpeech or LJSpeech

if dataset == 'LibriSpeech':
    LibriSpeech_dir = "../../../dataset/LibriSpeech/" # <replace with your LibriSpeech Location>
    sections = ['dev-clean/', 'test_clean/'] # <specify splits> 
    for section in sections:
        sub_secs = glob(LibriSpeech_dir + section + '*')
        for sub_sec in sub_secs:
            subsub_secs = glob(sub_sec + '/*')
            for subsub_sec in subsub_secs:
                audios.extend(glob(subsub_sec + '/*.flac')) 
else:
    LJ_dir = '../../../dataset/LJSpeech-1.1/wavs/' # <replace with your LJSpeech Location>
    LJ_audios=glob(LJ_dir + '*.wav')
    audios=LJ_audios

# read .flac and transform to ascii encode
from pydub import AudioSegment
import wave
from os.path import getsize
import numpy as np
from random import randint
from audioop import bias, lin2lin
from array import array
import io

audio_size = 0
audio_index = 0
audio_data = []
missed_bits = 0
raw_bytes = b''

# stipulate total data amount, train/val/test data amount
total_data = pow(2,30) # 1G
train_data = total_data / 16 # 64M
val_data = total_data / 64 # 16M
test_data = total_data - train_data - val_data


# simple progress bar
step = 5
progress = 0

# read all files
while audio_index < len(audios):
    audio = audios[audio_index]
    audio_index += 1
    
    # read by byte
    audio_segment = AudioSegment.from_file(audio)
    audio_segment.export('workspace.wav', format='wav')
    # already to wav
    with wave.open('workspace.wav') as f:
        # header, guess we can ignore
        metadata = f.getparams()
        # read bytes
        audio_frames = f.readframes(metadata.nframes)
        # convert to width 1
        # assume the compressed audios  have one channel
        audio_frames = bias(lin2lin(audio_frames, 2, 1), 1, 128)
        # for flac compression
        raw_bytes += audio_frames

        # every byte decode
        audio_frames = array('B', audio_frames).tolist()
        # right shift 1 bit
        audio_frames = [chr(x >> 1) for x in audio_frames]
        
        # every byte miss one bit
        # do not really store the missing bits here
        missed_bits += len(audio_frames) # number of bits
        audio_size += len(audio_frames) # number of bytes
        
        audio_data.extend(audio_frames)
    
    # progress
    if audio_index > len(audios) / 100 * progress:
        print('Progress %{}.'.format(progress))
        progress += step
    # use the first 256M
    if len(raw_bytes) >= total_data:
        break

print('Progress %100')

# cut the tail
raw_bytes = raw_bytes[:total_data]

# flac comression
with wave.open('workspace.wav', 'wb') as f:
    f.setnchannels(1)
    f.setsampwidth(1)
    f.setframerate(16000)
    f.writeframes(raw_bytes)
audio_segment = AudioSegment.from_file('workspace.wav', format='wav')
compressed = audio_segment.export('workspace.flac', format='flac', parameters=['-compression_level', '12']).read()
print(type(compressed))
print(len(compressed))

# cut audio_data every 2048 byte
start = 0
stride = 2048
audio_segments = []
while start < len(audio_data):
    end = min(start + stride, len(audio_data))
    # the segment
    segment = ''.join(audio_data[start:end])
    '''
    # random cut it into two pieces
    cut_point = randint(1, len(segment)-2)
    '''
    # use exactly half of the str to predict the other half
    cut_point = int(len(segment) / 2)
    # add a <s> in the front
    audio_segments.append('<s>' + segment[:cut_point] + '</s><s>' + segment[cut_point:] + '</s>')
    start = end

#constrain the number of samples
train = audio_segments[:train_data / pow(2,11)]
val = audio_segments[train_data / pow(2,11):train_data / pow(2,11) + val_data / pow(2,11)]
test = audio_segments[train_data / pow(2,11) + val_data / pow(2,11):]

def to_csv(list_of_str, file_name):
    import pandas as pd
    df = pd.DataFrame([[x] for x in list_of_str], columns=['text'])
    df.to_csv(file_name, escapechar='\\', index=False)

data_dir = './'
to_csv(train, data_dir + 'train.csv')
to_csv(val, data_dir + 'val.csv')
to_csv(test, data_dir + 'test.csv')
