CUDA_VISIBLE_DEVICES=0 \
python3 inference/colorization_pipeline.py \
    --input_file ./assets/test_images/malaysia19.jpg --output_file ./colorize_output/malaysia19.jpg \
    --model_path modelscope/damo/cv_ddcolor_image-colorization/pytorch_model.pt