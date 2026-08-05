config_path=configs/ar_diffusion_tf_chunkwise_5b_720p_only_HOIGen_dataset_new.yaml
logdir=logs/ar_diffusion_chunkwise_5b_720p_only_HOIGen_dataset_new

mkdir -p logdir
echo 'config_path'=$config_path
echo 'logdir'=$logdir

pkill -f "/m2v_intern/raozejing/StreamingCode/gpu.py"
bash /m2v_intern/raozejing/StreamingCode/train_gpu2.sh

torchrun --nproc_per_node=8 \
    train.py \
    --config_path $config_path \
    --logdir $logdir

bash /m2v_intern/raozejing/StreamingCode/train_gpu2.sh
bash /m2v_intern/raozejing/StreamingCode/train_gpu2.sh