Code for driving head motion with audio.

## Data

Configure voxceleb2 dataset as same as the global settings.

## Train

```shell
python train.py --rt_encode tc_no_lstm --audio_encode se --rt_decode l2l --batch_size 64 --drop 0.1 --name tmp --num_workers 4
```

Please note:

- Keep `num_workers` small because the dataloader need time to process.

## Test

Pretrained model can be downloaded from [Google Drive](https://drive.google.com/file/d/1ncEFOJuAqV-NM1ffa6uBFHZlulB8ItcC/view?usp=sharing). Put `save` folder into `./`.

```shell
python test.py --rt_encode tc_no_lstm --audio_encode se --rt_decode l2l --batch_size 4 --drop 0.1 --name tmp --num_workers 0
```