# # baseline
#######################
# 1
export dataset='cifar10'
# export out_dataset='SVHN'
export save=./checkpoint/$dataset/`(date "+%Y%m%d-%H%M%S")`'_cuda1__0drop_1.0sig_1.0ce_10.0ent_101epoch'/

mkdir -p $save
mkdir $save/copy
cp -r src $save/copy
cp -r scripts $save/copy

CUDA_VISIBLE_DEVICES=1 python2 src/main.py \
--out_folder $save \
--lr 0.1 \
--batch_size 64 \
--net_type wide-resnet \
--depth 28 --widen_factor 10 \
--dropout 0.0 \
--dataset $dataset \
\
--loss bce \
--bce_scale 1.0 \
--sharing 1.0 \
--ent 10.0 \
--pretrained 'wide-resnet' \
# --unknown_is_True True \
# --sampling_rate 0.3 \
# --resume
# --sepa_unknown_sharing True \
##########################