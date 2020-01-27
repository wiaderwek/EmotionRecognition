from movies_analysis import *
from music_generation import *
from music_dataset import *
import music_generation_constants as music_consts
import video_analysis_constants as video_consts
from moviepy.editor import VideoFileClip
import os
import math

def main():
    parser = argparse.ArgumentParser(description='Movies analysis.')
    parser.add_argument('--dir', type=str, help='Film directory')
    parser.add_argument('--name', type=str, help='Film name')
    args = parser.parse_args()

    ea = EmotionAnalyser()
    seq_length = 40
    image_shape = (80, 80, 3)
    #video_dir = "/home/tomasz/Dokumenty/shared/predict"
    video_dir = args.dir
    #video_name = "ACCEDE00018"
    video_name = args.name
    emotion_id, frames_num = ea.predict(40, (80, 80, 3), video_dir, video_name)
    emotion_hot = one_hot(emotion_id , music_consts.NUM_EMOTIONS)

    clip = VideoFileClip(os.path.join(video_dir, video_name + video_consts.VIDO_EXTENSION))
    bars = math.ceil(clip.duration/music_consts.BARS_PER_SECONDS)
    
    models = build_or_load()
    write_file('output', generate(models, bars, [emotion_hot]))
    
if __name__ == '__main__':
    main()