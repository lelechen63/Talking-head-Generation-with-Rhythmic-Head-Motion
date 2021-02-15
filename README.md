# Talking-head Generation with Rhythmic Head Motion (ECCV 2020)

Pytorch implementation for audio driven talking-head video synthesize. Given an inputed sampled video frames and a driving audio, our model makes use of 3D facial generation process to generate a head speaking the audio. Moreover, our model achieves controllable head motion as well as facial emotion, which results in more realistic talking-head video. We implement the model based on coding framework of [few-shot-vid2vid](https://github.com/NVlabs/few-shot-vid2vid).

## Results on VoxCeleb2 and Lip-reading-in-the-wild dataset
https://drive.google.com/drive/folders/1ApZwutK9aQYM6qGTCALp0IOEUyvU7ejt?usp=sharing

## Code Implementation

In this section, we will introduce how to implement our method. Including prerequest, dataset, training and testing.

### Prerequest

We run our code in Linux system with NVIDIA GPU and CUDA. To run the code, please prepare:

- Python 3
- Pytorch 1.3
- Numpy
- dlib
- dominate

### Dataset

We train and test our model in four datasets, [Crema](https://github.com/CheyneyComputerScience/CREMA-D), [Gride](https://www.grid.ac/downloads), [Voxceleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/), [Lrs3](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html) dataset.

For each of these datasets, we generate landmarks, 3D facial frames and calculate rotation of each targeted frames. The preprocess code will be released soon.

###  Training

[FlowNet2](https://github.com/NVIDIA/flownet2-pytorch) is applied in our model to guide the generation of optic flow between different frames. Compile flownet and download a pretrained weight with:

```
python scripts/download_flownet2.py
```

Taken Voxceleb as example, our model can be trained by running example code 

```
bash script/train_g8.sh
```

By this way, the model creates landmarks as intermediate generation based on audio input. And then it generates synthesized image by hybrid embedding module as well as nonlinear composition module, which takes both synthesized landmarks, generated 3D projection facial image, and sampled video frames as input.

We also apply multiple choices to train your own model. The entire example bash code is shown as below:

```
CUDA_VISIBLE_DEVICES=[CUDA Ids] python train.py \
--name face8_vox_new \
--dataset_mode facefore \
--adaptive_spade \
--warp_ref \
--warp_ani \
--spade_combine \
--add_raw_loss \
--gpu_ids [Gpu Ids] \
--batchSize 4 \
--nThreads 8 \
--niter 1000 \
--niter_single 1001 \
--n_shot 8 \
--n_frames_G 1 \
--dataroot 'voxceleb2' \
--dataset_name vox \
--save_epoch_freq 1 \
--display_freq 5000 \
--continue_train \
--use_new \
--crop_ref
```

- To indicate directory of model, please sepecify its name in `--name` which will be created in `checkpoints`. Moreover, if `--continue_train` is provided, model will load pretrained weight from directory described by `--name`. User can also use `--dataset_mode` to define dataloader to be applied. For example `--dataset_mode facefore` indicates `data/facefore_dataset.py` will be exploited.
- In order to control general training process, please use `--batchSize` for number of training example for each epoch, `--niter` to set total number of epoches of training. Note that our model has the capacity to support temporal training in the future, `--n_frames_G` refers to number of images to be used for each time slice, and `--niter_single` indicates in which epoch the model starts to trained temporally. (Currently, we only support non-temporal training).
- For purpose of ablation study, you can modify the model with `--warp_ref`, `--warp_ani`, which refers to whether compose synthesized image with warpped sampled image or 3D facial image specifically. If specific `--spade_combine`, the model will be train with nonlinear composed module. Otherwise, linear composed module will be applied. If specific `--use_new`, hybrid embedding module and a noval composition method will be contained in the model, otherwise, method similar to [few-shot-vid2vid](https://github.com/NVlabs/few-shot-vid2vid) will be exploited. Furthermore, please use `--no_warp` for composition method with non-warpping (original) images, `--no_atten` for mean-embeding rather than hybrid embedding.

### Testing

After training, you can test results on datasets (e.g. voxceleb) by using following simple script:

```
bash test_demo.sh
```

The script will load videos in `demo` directory and synthesized videos by take several sampled frames as reference from related one. Required file for each video will be disucss later.

In order to test self-decide model, the detail script is shown as below:

```
CUDA_VISIBLE_DEVICES=[CUDA Ids] python test_demo_ani.py \
--name face8_vox_new \
--dataset_mode facefore_demo \
--adaptive_spade \
--warp_ref \
--warp_ani \
--add_raw_loss \
--spade_combine \
--example \
--n_frames_G 1 \
--which_epoch latest \
--how_many 10 \
--nThreads 0 \
--dataroot 'demo' \
--ref_img_id "0" \
--n_shot 8 \
--serial_batches \
--dataset_name vox \
--crop_ref \
--use_new
```

Except same flags as training, you can use `--ref_img_id` to indicates index of sample frames, and `--n_shot` for number of reference images used during training.

For demo testing, several files about video need to be provided in `demo` directory.

- Reference video and landmarks. For example, `00181_aligned.mp4` and `00181_aligned.npy`. 
- Animation video. 3D animation video generated by 3D generator needed to be pre-provided. If `--warp_ani` is not specified, animation video is not necessary. (e.g. `00181_aligned_ani.mp4`)
- Rotation of face in the video. Rotation of face in each frame of video. In our demo, it refers to file `00181_aligned_rt.npy`.
- Front face image and relate landmark. Frame id of front face (rotation closest to 0) in video is needed to be provided. Related landmarkd will be applied to create landmarks of animation video during data loading. For example, in `00181_aligned.mp4`, front face id is 234, and landmark refers to `00181_aligned_front.py`. 
- Audio. If video needed to be synthesized from audio, related voice need to be provided. Otherwise, landmarks of target image should be provided.

Moreover, pretrained weight should be placed in directory sepecified by `--name`. For example, `checkpoints/face8_vox_demo` in our demo.

### Pretrained weight

Our pretrained weight can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1JbQhnNyHBbYtikg5S_B5oS_vk81j3oPT?usp=sharing/).

## Citation

    @article{chen2020talking,
      title={Talking-head Generation with Rhythmic Head Motion},
      author={Chen, Lele and Cui, Guofeng and Liu, Celong and Li, Zhong and Kou, Ziyi and Xu, Yi and Xu, Chenliang},
      journal={arXiv preprint arXiv:2007.08547},
      year={2020}
    }

## Future

More detail will be update in following days.
