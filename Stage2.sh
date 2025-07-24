#export CUDA_VISIBLE_DEVICES=0,1
#
export PYTHONPATH=$(pwd):$PYTHONPATH

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=8 \
  --master_port=34213 \
  swift_grpo.py \
  --rlhf_type grpo \
  --external_plugins plugin.py \
  --system sysprompt.txt \
  --model_type custom \
  --vllm_limit_mm_per_prompt '{"image":2}' \
  --reward_funcs external_cls_acc_choice external_cosine format repetition \
  --reward_weights 3 1 1 1 \
  --max_completion_length 512 \
  --soft_cache_length 256 \
  --freeze_llm false \
  --freeze_vit false \
  --freeze_aligner false \
  --use_vllm false \
  --log_completions true \
  --vllm_device auto \
  --vllm_gpu_memory_utilization 0.6 \
  --train_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 4 \
  --learning_rate 1e-5 \
  --gradient_accumulation_steps 2 \
  --save_strategy steps \
  --eval_strategy no \
  --save_steps 500 \
  --save_total_limit 10 \
  --logging_steps 1 \
  --warmup_ratio 0.01 \
  --dataloader_num_workers 4 \
  --num_generations 8 \
  --dynamic_sample true \
  --temperature 1 \
  --report_to tensorboard \
  --num_iterations 1 \
  --num_infer_workers 4 \
  --async_generate false \
  --beta 5e-3 \
  --model_kwargs '{"MAX_NUM":6, "dynamic_image_size":true, "neighbor_dis":6}' \
  --modules_to_save mlp1 projector soft_prompt \
  --deepspeed zero0 \
  --max_resample_times 3 \
  --repetition_n_grams 4 \
  --model stage1_outputs/v1-20250701-172230/checkpoint-6000 \
  --dataset /mnt/vlr/laishi/train_stage2_data.jsonl \
  --output_dir stage2_outputs \
  # --resume_from_checkpoint stage2_outputs/v1-20250720-000008/checkpoint-5000 \
  # --resume_only_model true