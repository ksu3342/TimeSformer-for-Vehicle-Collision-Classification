import os

for filename in os.listdir('/home/sunyvbo/demo1/dataset/test/CarCollisionDataset'):
    if filename.endswith('.mp4'):
        input_file = os.path.join('/home/sunyvbo/demo1/dataset/test/CarCollisionDataset', filename)
        output_file = os.path.join('/home/sunyvbo/demo1/dataloader/Modifiedvideo', filename)
        os.system(f'ffmpeg -i {input_file} -ss 00:00:06 -t 00:00:04 -r 25 {output_file}')
