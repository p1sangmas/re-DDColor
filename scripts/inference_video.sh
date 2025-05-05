CUDA_VISIBLE_DEVICES=0 \
python3 inference/colorization_pipeline_video.py \
    --video_path ./video_input/merdeka.mp4 \
    --model_path modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt \
    --output_video_path ./video_output/merdeka_colorized.mp4