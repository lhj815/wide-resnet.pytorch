# # baseline
export dataset='cifar100'
# export out_dataset='SVHN'
export save=./checkpoint/$dataset/`(date "+%Y%m%d-%H%M%S")`'_cuda1_resnet34_softmax_batch128'/
mkdir -p $save
mkdir $save/copy
cp -r src $save/copy
cp -r scripts $save/copy

CUDA_VISIBLE_DEVICES=1 python2 src/main_cifar100.py \
--batch_size 128 \
--out_folder $save \
--lr 0.1 \
--net_type resnet_mahala \
--dropout 0.0 \
--dataset $dataset \
\
--loss ce \
# --bce_scale 1.0 \
# --sharing 1.0 \
# --ent 0.5 \
# --sampling_rate 0.2 \
# --resume
# --unknown_is_True True \
# --sepa_unknown_sharing True \