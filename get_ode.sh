config_path="configs/causal_ode_chunkwise_5b_720P.yaml"
generator_ckpt="logs/ar_diffusion_chunkwise_5b_720p_only_HOIGen_dataset_new_bs32/checkpoint_model_016000/model.pt"
rawdata_path="/m2v_intern/raozejing/StreamingCode/dataset/live_5s_pose_1_2_3_4_5_6_7_8_9_10_11_rzj_without_fabric_and_keep_half_shoes/split_5.csv"
output_folder="/m2v_intern_v3/raozejing/dataset/ODE_Causal_chunkwise_latents_only_origin_dataset_new_run_0519"
prefix="5"

echo "config_path: $config_path"
echo "generator_ckpt: $generator_ckpt"
echo "rawdata_path: $rawdata_path"
echo "output_folder: $output_folder"
echo "prefix: $prefix"

ps -ef | grep /home/raozejing/gpu.py | grep -v grep | awk '{print $2}' | xargs kill
pkill -f "/m2v_intern/raozejing/StreamingCode/gpu.py"
bash /m2v_intern/raozejing/StreamingCode/train_gpu2.sh

torchrun \
    --nproc_per_node=8 \
    get_causal_ode_data_chunkwise.py \
    --config_path $config_path \
    --generator_ckpt $generator_ckpt \
    --rawdata_path $rawdata_path \
    --output_folder $output_folder \
    --prefix $prefix \
    --guidance_scale 5.0

bash /m2v_intern/raozejing/StreamingCode/train_gpu2.sh
bash /m2v_intern/raozejing/StreamingCode/train_gpu2.sh