# Data Prerprocessing for one video:

NOTE: The video path need to be absolute path

 - > Install face_alignment package:
   

    pip install face-alignment

 or    

    conda install -c 1adrianb face_alignment

   
 - >Step1: We first extract the facial landmarks of a video.

     cd data

- Process single video file:

    python single_video_preprocess.py --extract_landmark --video_path=/a/b/c.mp4

 - Or process all videos in a folder: 

     python single_video_preprocess.py --extract_landmark --video_path=/a/b
- >  Step 2: Use RT_compute to compute the RT betwen the canonical landmark and target landmark：
- process single video file:

    python single_video_preprocess.py --compute_rt --video_path=/a/b/c.mp4
- Process all videos in a folder: 

    python single_video_preprocess.py --compute_rt --video_path=/a/b

- > Step 3: Find the most frontalized frame

    python single_video_preprocess.py --get_front  --video_path=/a/b/c.mp4
- > Step 4: Then we need to render the 3D image using the RT and the most fronalized image frame

    cd PRNet
    
    python get_3d.py --img_path /a/b/c__00068.png   # the image path should be the one generated in setp3.

- > Step5: Generate 3D rendering video

    cd face_tool
    
    python find_camera.py --front_img_path /a/b/c__00068.png
    
    python find_camera.py --front_img_path /u/lchen63/github/Talking-head-Generation-with-Rhythmic-Head-Motion/test_data/sample3/ref_id_crop.png --prnet_lmark_path /u/lchen63/github/Talking-head-Generation-with-Rhythmic-Head-Motion/test_data/sample3/ref_id_crop__prnet.npy --front_lmark_path  /u/lchen63/github/Talking-head-Generation-with-Rhythmic-Head-Motion/test_data/sample3/Pose__front.npy --ref_lmark_path /u/lchen63/github/Talking-head-Generation-with-Rhythmic-Head-Motion/test_data/sample3/ref_id_crop__front.npy
