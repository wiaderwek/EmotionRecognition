from movies_analysis import *
from music_generation import *
from datasets.music_dataset import *
import constants.music_generation_constants as music_consts
import constants.video_analysis_constants as video_consts
from moviepy.editor import VideoFileClip
import os
import math

def main():
    parser = argparse.ArgumentParser(description='Movies analysis.')
    parser.add_argument('--dir', type=str, help='Film directory')
    parser.add_argument('--name', type=str, help='Film name')
    parser.add_argument('--out', type=str, help='Out file name')
    args = parser.parse_args()
    
    assert len(args.dir) > 0
    assert len(args.name) > 0
    assert len(args.out) > 0

    ea = EmotionAnalyser()
    video_dir = args.dir
    video_name = args.name
    emotion_id, frames_num = ea.predict(video_consts.SEQ_LENGTH, video_consts.IMG_SHAPE, video_dir, video_name)
    emotion_hot = one_hot(emotion_id , music_consts.NUM_EMOTIONS)

    clip = VideoFileClip(os.path.join(video_dir, video_name + video_consts.VIDO_EXTENSION))
    bars = math.ceil(clip.duration/music_consts.BARS_PER_SECONDS)
    
    models = build_or_load()
    write_file(args.out, generate(models, bars, [emotion_hot]))
    
if __name__ == '__main__':
    main()