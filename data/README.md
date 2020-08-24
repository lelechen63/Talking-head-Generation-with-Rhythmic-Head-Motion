 
 - >Step1: We first extract the facial landmarks of a video.

     cd data

- Process single video file:

    python single_video_preprocess.py --extract_landmark --video_path=/a/b/c.mp4

 - Or process all videos in a folder: 

     python single_video_preprocess.py --extract_landmark --video_path=/a/b
- >  Use RT_compute to compute the RT betwen the canonical landmark and target landmark：
- process single video file:

    python single_video_preprocess.py --compute_rt --video_path=/a/b/c.mp4
- Process all videos in a folder: 

    python single_video_preprocess.py --compute_rt --video_path=/a/b

- > Find the most frontalized frame

    python single_video_preprocess.py --get_front  --video_path=/a/b/c.mp4
- > Then we need to render the 3D image using the RT and the most fronalized image frame
   cd 
