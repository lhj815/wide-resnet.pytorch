export check_point='1'
export dataset='cifar100'
export out_dataset='Tiny'
export mode='softmax'
export outf='./results/test_detection'/$dataset/$out_dataset'_'`(date "+%Y%m%d-%H%M%S")`'_'$mode'_'$check_point
# export input_preproc_noise_magni=0.0014
# export outf='./results/test_detection'/$dataset/$out_dataset'_'`(date "+%Y%m%d-%H%M%S")`'_'$mode'_'$check_point'_'$input_preproc_noise_magni/
# export odin=1000
# export outf='./results/test_detection'/$dataset/$out_dataset'_'`(date "+%Y%m%d-%H%M%S")`'_'$mode'_'$check_point'_'$odin'_'$input_preproc_noise_magni/




CUDA_VISIBLE_DEVICES=1 python2 src/test_detection.py \
--outf $outf \
--net_type wide-resnet \
--depth 28 \
--widen_factor 10 \
--dropout 0.3 \
\
--dataset $dataset \
--out_dataset $out_dataset \
--mode $mode \
--check_point $check_point \
# --input_preproc_noise_magni $input_preproc_noise_magni \
# --odin $odin \