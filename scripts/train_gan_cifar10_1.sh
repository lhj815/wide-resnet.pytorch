# # baseline
#######################
# 1
export dataset='cifar10'
# export out_dataset='SVHN'
export save=./checkpoint/$dataset/`(date "+%Y%m%d-%H%M%S")`'_cuda1__1.0sig__gan_lr2e-5_adam'/
mkdir -p $save
mkdir $save/copy
cp -r src $save/copy
cp -r scripts $save/copy

CUDA_VISIBLE_DEVICES=1 python2 src/main_gan.py \
--out_folder $save \
--lr 2e-5 \
--net_type wide-resnet \
--depth 28 --widen_factor 10 \
--dropout 0.3 \
--dataset $dataset \
\
--loss bce \
--bce_scale 1.0 \
--gan True \
# --fake_node_bce_beta 0.2 \
# --sharing 0.5 \
# --ent 0.5 \

# --sampling_rate 0.3 \
# --resume
# --unknown_is_True True \
# --sepa_unknown_sharing True \
##########################