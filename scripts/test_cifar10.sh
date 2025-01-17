# export check_point='20190808-001511_cuda1__0drop_softmax_101epoch_95.56%'
# export check_point='20190804-180528_cuda0_1.0sig_1.0ce_10.0ent_94.72%'
export check_point='20190807-000929_cuda0__0.0sig_1.0ce_5.0ent_101epoch_95.15%'
# 20190803-213245_cuda0__1.0sig_1.0ce_5.0ent_difficult_95.11%
# 20190803-220017_cuda0_0.0sig_1.0ce_10.0ent_difficult_94.30%
# 20190722-135525_1.0sig_0.0ce_10.0ent_92.18%
# export check_point='20190722-135525_1.0sig_0.0ce_10.0ent_92.18%'

export dataset='cifar10'
export out_dataset='Tiny'
# export out_dataset='Tiny'
# export mode='only_sharing_node'
export mode='sharing_include_softmax'
# sharing_node_print_include_softmax
export outf='./results/test_detection'/$dataset/$out_dataset'_'`(date "+%Y%m%d-%H%M%S")`'_'$mode'_'$check_point
# export input_preproc_noise_magni=0.1
# export outf='./results/test_detection'/$dataset/$out_dataset'_'`(date "+%Y%m%d-%H%M%S")`'_'$mode'_'$check_point'_'$input_preproc_noise_magni/
# export odin=1000
# export outf='./results/test_detection'/$dataset/$out_dataset'_'`(date "+%Y%m%d-%H%M%S")`'_'$mode'_'$check_point'_'$input_preproc_noise_magni'_'$odin/



# CUDA_LAUNCH_BLOCKING=1
CUDA_VISIBLE_DEVICE=1 python src/test_detection.py \
--outf $outf \
--net_type wide-resnet \
--batch_size 100 \
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