CUDA_VISIBLE_DEVICES=0 python train_tacred.py \
  --model_name_or_path roberta-large \
  --input_format entity_marker_punct \
  --seed 78 \
  --run_name causal \
  --output_dir outputs/exp \
  --k_size 3 
