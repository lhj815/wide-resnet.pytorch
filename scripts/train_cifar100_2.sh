# # baseline
export dataset='cifar100'
# export out_dataset='SVHN'
export save=./checkpoint/$dataset/`(date "+%Y%m%d-%H%M%S")`'_cuda1_1.0bce_1.0ce_5.0ent_lr_1e-4_sampling_rate0.2'/
mkdir -p $save
mkdir $save/copy
cp -r src $save/copy
cp -r scripts $save/copy

CUDA_VISIBLE_DEVICES=0 python2 src/main.py \
--out_folder $save \
--lr 1e-4 \
--net_type wide-resnet \
--pretrained 'wide-resnet' \
--depth 28 --widen_factor 10 \
--dropout 0.3 \
--dataset $dataset \
\
--loss bce \
--sharing 1.0 \
--ent 5.0 \
--bce_scale 1.0 \
--sampling_rate 0.2 \
# --sigmoid_sum 1.0 \
# --resume
# --unknown_is_True True \
# --sepa_unknown_sharing True \