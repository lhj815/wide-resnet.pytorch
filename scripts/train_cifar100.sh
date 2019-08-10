# # baseline
export dataset='cifar100'
# export out_dataset='SVHN'
export save=./checkpoint/$dataset/`(date "+%Y%m%d-%H%M%S")`'_cuda1_resnet34_1.0sig_1.0ce_0.5ent_sampling_rate0.2_batch128'/
mkdir -p $save
mkdir $save/copy
cp -r src $save/copy
cp -r scripts $save/copy

CUDA_VISIBLE_DEVICES=1 python2 src/main.py \
--out_folder $save \
--lr 0.1 \
--net_type wide-resnet \
--pretrained 'resnet34' \
--depth 28 --widen_factor 10 \
--dropout 0.3 \
--dataset $dataset \
\
--loss bce \
--sharing 1.0 \
--ent 0.5 \
--bce_scale 1.0 \
--sampling_rate 0.2 \
# --resume
# --unknown_is_True True \
# --sepa_unknown_sharing True \