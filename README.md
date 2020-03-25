# Tilt Your Head When “You” are Talking

Pytorch implementation for audio driven talking-head video synthesize. Given an inputed sampled video frames and a driving audio, our model makes use of 3D facial generation process to generate a head speaking the audio. Moreover, our model achieves controllable head motion as well as facial emotion, which results in more realistic talking-head video. We implement the model based on coding framework of [few-shot-vid2vid](https://github.com/NVlabs/few-shot-vid2vid).


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
bash train_g8.sh
```

By this way, the model creates landmarks as intermediate generation based on audio input. And then generate synthesized image by hybrid embedding module as well as nonlinear composition module, which takes both synthesized landmarks, generated 3D projection facial image, and sampled video frames as input.

We also apply multiple choices to train your own model. The entire example bash code is shown as below:

```
CUDA_VISIBLE_DEVICES=[Cuda Ids] python train.py --name face8_vox_ani_nonlinear_continue --dataset_mode facefore \
--adaptive_spade --warp_ref --warp_ani --spade_combine --add_raw_loss \
--gpu_ids [Cuda Indexs] --batchSize 56 --nThreads 64 --niter 100 --niter_single 100 \
--n_shot 8 --n_frames_G 1 \
--dataroot '/mnt/Data/lchen63/voxceleb2' --dataset_name vox --save_epoch_freq 1 --display_freq 1000 \
--continue_train --crop_ref --use_new
```

- In order to control general training process, please use `--batchSize` to number of training example for each epoch, `--niter` to set total number of epoches of training. Note that our model has the capacity to support temporal training in the future, `--n_frames_G` refers to number of images to be used for each time slice, and `--niter_single` indicates in which epoch the model starts to trained temporally. (Currently, we only support non-temporal training).
- For purpose of ablation study, you can modify the model with `--warp_ref`, `--warp_ani`, which refers to whether compose synthesized image with warpped sampled image or 3D facial image specifically. If specific `--spade_combine`, the model will be train with nonlinear composed module. Otherwise, linear composed module will be applied. If specific `--use_new`, hybrid embedding module and a noval composition method will be contained in the model, otherwise, method similar to few-shot-vid2vid will be exploited. Furthermore, please use `--no_warp` for composition method with non-warpping (original) images, `--no_atten` for mean-embeding rather than hybrid embedding.

### Testing

After training, you can test results on datasets (e.g. voxceleb) by using following simple script:

```
bash test_demo.sh
```

The script will randomly select several videos from voxceleb and synthesized videos by take several sampled frames from it.

In order to test self-decide model, the detail script is shown as below:

```
CUDA_VISIBLE_DEVICES=[cuda Ids] python test_demo.py --name face8_vox_ani \
--dataset_mode facefore_demo \
--adaptive_spade \
--warp_ref \
--warp_ani \
--example \
--n_frames_G 1 \
--which_epoch latest \
--how_many [number of test videos] \
--nThreads 0 \
--dataroot '/home/cxu-serve/p1/common/voxceleb2' \
--ref_img_id "0" \
--n_shot 1 \
--serial_batches \
--dataset_name vox \
--crop_ref \
--use_new
```

Except similar flags as training, you can use `--ref_img_id` to indicates index of sample frames, and `--n_shot` to specific number of sample frames.

For demo testing, you can run the following example script:

```
bash test_demo_example.sh
```

The detail is shown below:

```
CUDA_VISIBLE_DEVICES=$1 python test_demo_example.py --name face8_vox_ani_nonlinear \
--dataset_mode facefore_demo \
--adaptive_spade \
--warp_ref \
--warp_ani \
--add_raw_loss \
--spade_combine \
--example \
--n_frames_G 1 \
--which_epoch $2 \
--how_many $3 \
--nThreads 0 \
--dataroot '/home/cxu-serve/p1/common/voxceleb2' \
--ref_dataroot '/home/cxu-serve/p1/common/voxceleb2' \
--ref_img_id "0" \
--n_shot 1 \
--serial_batches \
--dataset_name vox \
--tgt_video_path "/home/cxu-serve/p1/common/voxceleb2/unzip/test_video/id00017/01dfn2spqyE/00001_aligned.mp4" \
--ref_dataset vox \
--ref_video_path "/home/cxu-serve/p1/common/voxceleb2/unzip/test_video/id00017/01dfn2spqyE/00001_aligned.mp4" \
--ref_ani_id 10 \
--finetune
```

`--dataset_name`, `--tgt_video_path` and `--dataroot` refers to directory of target video, while `--ref_dataset`, `--ref_video_path` and `--ref_dataroot` refers to directory of reference video. You can use `--ref_img_id` to select specific frames from reference video as sample images. More related path can be seen in code of "test_demo_example.py", and we encourage user to read that.

## Future

Demo test method and more detail will be update in following days.