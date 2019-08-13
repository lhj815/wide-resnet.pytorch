# # baseline
export dataset='cifar100'
# export out_dataset='SVHN'
# export save=./checkpoint/$dataset/`(date "+%Y%m%d-%H%M%S")`'gpu1_softmax_ent1.0_300epoch'/
export save=./checkpoint/$dataset/`(date "+%Y%m%d-%H%M%S")`'gpu1_soft_5.0sharing_50ent_300epoch_all_logit'/

mkdir -p $save
mkdir $save/copy
cp -r src $save/copy
cp -r scripts $save/copy

CUDA_VISIBLE_DEVICES=1 python2 src/main_cifar100.py \
--out_folder $save \
--lr 0.1 \
--net_type wide-resnet \
--depth 28 --widen_factor 10 \
--dropout 0.0 \
--dataset $dataset \
\
--loss ce \
--sharing 5.0 \
--ent 50.0 \
# --ce_entropy 1.0 \
# --sampling_rate 0.1 \
# --sigmoid_sum 1.0 \
# --resume
# --unknown_is_True True \
# --sepa_unknown_sharing True \