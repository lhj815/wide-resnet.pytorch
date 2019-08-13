# export check_point='20190810-233740gpu1_softmax_ent0.1_300epoch_81.41%'
export check_point='20190812-190914gpu0_ce_bce__1.0ce_5.0bce_300epoch_20cp_80.95%'

export dataset='cifar100'
export out_dataset='Tiny'
export mode='softmax'
export batch_size=100
export outf='./results/test_detection'/$dataset/$out_dataset'_'`(date "+%Y%m%d-%H%M%S")`'_'$mode'_'$check_point
# export input_preproc_noise_magni=0.1
# export batch_size=2
# export outf='./results/test_detection'/$dataset/$out_dataset'_'`(date "+%Y%m%d-%H%M%S")`'_'$mode'_'$check_point'_batch'$batch_size'_'$input_preproc_noise_magni/
# export odin=1000
# export outf='./results/test_detection'/$dataset/$out_dataset'_'`(date "+%Y%m%d-%H%M%S")`'_'$mode'_'$check_point'_batch'$batch_size'_'$odin'_'$input_preproc_noise_magni/




CUDA_VISIBLE_DEVICES=1 python2 src/test_detection.py \
--batch_size $batch_size \
--outf $outf \
--net_type wide-resnet \
--depth 28 \
--widen_factor 10 \
--dropout 0.0 \
\
--dataset $dataset \
--out_dataset $out_dataset \
--mode $mode \
--check_point $check_point \
# --input_preproc_noise_magni $input_preproc_noise_magni \
# --odin $odin \