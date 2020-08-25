 Install face_alignment package:
   pip install face-alignment	
 or 
   conda install -c 1adrianb face_alignment
 If it is not working, please check https://github.com/1adrianb/face-alignment for details.
 
 - >Step1: We first extract the facial landmarks of a video.

     cd data

- Process single video file:

    python single_video_preprocess.py --extract_landmark --video_path=/a/b/c.mp4

 - Or process all videos in a folder: 

     python single_video_preprocess.py --extract_landmark --video_path=/a/b
- > Step2: Use RT_compute to compute the RT betwen the canonical landmark and target landmark：
- process single video file:

    python single_video_preprocess.py --compute_rt --video_path=/a/b/c.mp4
- Process all videos in a folder: 

    python single_video_preprocess.py --compute_rt --video_path=/a/b

- > Step3: Find the most frontalized frame

    python single_video_preprocess.py --get_front  --video_path=/a/b/c.mp4
- > Step 4 Then we need to render the 3D image using the RT and the most fronalized image frame
   I will post this later...
