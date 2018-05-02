VAEWGAN
===



## Usage


### Training

python train.py --model=vaewgan --dataset=datasets/files/celebA.hdf5 --epoch=1000 --batchsize=100 --output=output --resume=False

### Testing

python test.py --model=cvaewgan --dataset=datasets/files/celebA.hdf5 --epoch=1000 --batchsize=100 --output=output --resume=True
