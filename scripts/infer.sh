# baseline
export dataset='cifar10'
export out_dataset='SVHN'
export save=./checkpoint/$dataset/`(date "+%Y%m%d-%H%M%S")`/
mkdir -p $save
mkdir $save/copy
cp -r src $save/copy
cp -r scripts $save/copy

python2 src/main.py \
--lr 0.01 \
--net_type wide-resnet \
--depth 28 --widen_factor 10 \
--dropout 0.3 \
--dataset $dataset \
--out_dataset $out_dataset \
\
--loss bce \
--sharing 1.0 \
--ent 20.0 \
--unknown_is_True True \
--sepa_unknown_sharing True \
--resume \
--inferOnly True
