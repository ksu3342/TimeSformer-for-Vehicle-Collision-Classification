import cv2
import os

def print_video(video_path):
    video_files = [f for f in os.listdir(video_path) if f.endswith('.mp4')]

    i=0
    for video_file in video_files:
        cap = cv2.VideoCapture(os.path.join(video_path,video_file))

        frame_count = int (cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int (cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int (cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f'Video {video_file}:')
        print(f"Frame count:{frame_count}")
        print(f"Frame width:{frame_width}"+"    "+f"Frame height:{frame_height}")

        cap.release()
        i=i+1
        print("No."+str(i))

print_video('/home/yanjiawen/ProjectTwo/dataset/TEXT/0')
