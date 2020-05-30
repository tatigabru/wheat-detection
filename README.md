# Wheat Detection Challenge


## Make folds
python -m src.folds.make_folds

## Test the train runners
python -m src.pretrain_runner --model-name "unet_se_resnext101_32x4d" --encoder "se_resnext101_32x4d" --debug True --image-size 224 --epochs 2 --lr 1e-3 --batch-size 16 --num-workers 2

python -m src.train_runner --model-name "unet_se_resnext101_32x4d" --encoder "se_resnext101_32x4d" --debug True --image-size 224 --epochs 2 --lr 1e-3 --batch-size 16 --num-workers 2 

