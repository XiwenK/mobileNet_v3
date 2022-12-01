# MobileNetV3_Large Based Improvements

## Environment setup
```shell
module load anaconda # Server Only
conda create -n mmdl python=3.8
conda activate mmdl
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

pip install -U openmim
mim install mmcv-full

git clone https://github.com/open-mmlab/mmclassification.git
cd mmclassification
pip install -v -e .
pip install tensorboard
```

## Run training
### Local
```shell
python tools/train.py configs/mobilenet_v3/mobilenet_v3_small_b128_cifar10.py
```

### On Server
```shell
sbatch tools/server_train.sh configs/mobilenet_v3/mobilenet_v3_small_b128_cifar10.py
```

## Draw the training graph
```shell
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/xxx directory/xxx.log.json --keys train_accuracy accuracy_top-1 --title "xxx" --legend train val --out xxx.jpg 

# Example
python tools/analysis_tools/analyze_logs.py plot_curve work_dirs/mobilenet_v3_large_b128_cifar100/20221120_045648.log.json --keys train_accuracy accuracy_top-1 --title "Baseline MobileNetV3_Large on CIFAR100" --legend train val --out baseline_cifar100.jpg 
```

## Evaluation Results

### Baseline
+ mobilenetv3_large_b128_cifar10:
    + Accuracy: 94.02
    + Logs: work_dirs/mobilenet_v3_large_b128_cifar10/20221119_224117.log
+ mobilenet_v3_large_b128_cifar100:
    + Accuracy: 75.47
    + Logs: work_dirs/mobilenet_v3_large_b128_cifar100/20221110_222220.log 

### Data Augument
+ cutmix_mobilenet_v3_large_b128_cifar10:
    + Accuracy: 95.18999
    + Logs: work_dirs/cutmix_mobilenet_v3_large_b128_cifar10/20221120_091955_cutmix.log
+ cutmix_mobilenet_v3_large_b128_cifar100:
    + Accuracy: 79.5400
    + Logs: work_dirs/cutmix_mobilenet_v3_large_b128_cifar100/20221116_165504_cutmix.log

### Training Strategy
+ poly_warmup5_mobilenet_v3_large_b128_cifar10:
    + Accuracy: 94.539
    + Logs: work_dirs/poly_warmup5_mobilenet_v3_large_b128_cifar10/poly_warmup5.log
+ poly_warmup5_mobilenet_v3_large_b128_cifar100:
    + Accuracy: 75.50999
    + Logs: work_dirs/poly_warmup5_mobilenet_v3_large_b128_cifar100/poly_warmup5_large.log

### Model Modification
+ bsconvs_mobilenet_v3_large_b128_cifar10:
    + Accuracy: 94.12
    + Logs: work_dirs/bsconvs_mobilenet_v3_large_b128_cifar10/20221120_141652.log
+ bsconvs_mobilenet_v3_large_b128_cifar100:
    + Accuracy: 76.41
    + Logs: work_dirs/bsconvs_mobilenet_v3_large_b128_cifar100/20221120_120748.log

### Overall Evaluation
+ 1.cifar10_data_with_bsconv
    + Accuracy:94.82999
    + Logs:work_dirs/1.cifar10_data_with_bsconv/20221120_160426_cifar10_bsconv.log

+ 2.cifar10_data_with_poly
    + Accuracy:95.61
    + Logs:work_dirs/2.cifar10_data_with_poly/20221120_235117_cifar10_poly.log

+ 3.cifar10_bsconv_with_poly
    + Accuracy:94.61
    + Logs:work_dirs/3.cifar10_bsconv_with_poly/20221120_235711.log

+ 4.cifar10_all_together
    + Accuracy:95.28
    + Logs:work_dirs/4.cifar10_all_together/20221121_024031.log

+ 5.cifar100_data_with_bsconv
    + Accuracy:79.45
    + Logs:work_dirs/5.cifar100_data_with_bsconv/20221121_000725.log

+ 6.cifar100_data_with_poly
    + Accuracy:80.24
    + Logs:work_dirs/6.cifar100_data_with_poly/20221121_021603.log

+ 7.cifar100_bsconv_with_poly
    + Accuracy:77.81
    + Logs:work_dirs/7.cifar100_bsconv_with_poly/20221121_000402.log

+ 8.cifar100_all_together
    + Accuracy:80.9
    + Logs:work_dirs/8.cifar100_all_together/20221121_040730.log
