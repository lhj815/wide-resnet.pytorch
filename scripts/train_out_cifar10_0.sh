# # baseline
#######################
# 1
export dataset='cifar10'
export out_dataset='Tiny'
# export out_dataset='SVHN'
export save=./checkpoint/$dataset/`(date "+%Y%m%d-%H%M%S")`'_out_cuda0__0.0sig_1.0ce_5.0ent_101epoch'/

mkdir -p $save
mkdir $save/copy
cp -r src $save/copy
cp -r scripts $save/copy

CUDA_VISIBLE_DEVICES=0 python2 src/main_train_out.py \
--resume \
--out_folder $save \
--lr 0.0001 \
--batch_size 64 \
--net_type wide-resnet \
--depth 28 --widen_factor 10 \
--dropout 0.3 \
--dataset $dataset \
--out_dataset $out_dataset \
\
--loss bce \
--bce_scale 0.0 \
--sharing 1.0 \
--ent 5.0 \
# --unknown_is_True True \
# --sampling_rate 0.3 \

# --sepa_unknown_sharing True \
##########################

