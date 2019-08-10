# # baseline
#######################
# 1
export dataset='cifar10'
export out_dataset='Tiny'
export save=./checkpoint/$dataset/`(date "+%Y%m%d-%H%M%S")`'_cuda1__AnotherNet_1.0sig_1.0ce_5.0ent_101epoch'/

mkdir -p $save
mkdir $save/copy
cp -r src $save/copy
cp -r scripts $save/copy

CUDA_VISIBLE_DEVICES=1 python2 src/main_another_net.py \
--out_folder $save \
--lr 0.1 \
--batch_size 64 \
--net_type wide-resnet \
--depth 28 --widen_factor 10 \
--dropout 0.3 \
--dataset $dataset \
--out_dataset $out_dataset \
\
--loss bce \
--bce_scale 1.0 \
--sharing 1.0 \
--ent 5.0 \
# --unknown_is_True True \
# --sampling_rate 0.3 \
# --resume

# --sepa_unknown_sharing True \
##########################