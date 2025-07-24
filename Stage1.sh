#export CUDA_VISIBLE_DEVICES=0,1
#
export PYTHONPATH=$(pwd):$PYTHONPATH

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=8 \
  --master_port=34229 \
  swift_sft.py \
  --num_train_epochs 2 \
  --model_type custom \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --model_kwargs '{"MAX_NUM":6, "dynamic_image_size":true, "neighbor_dis":6}' \
  --freeze_llm true \
  --freeze_vit true \
  --freeze_aligner false \
  --train_type full \
  --torch_dtype bfloat16 \
  --learning_rate 1e-5 \
  --weight_decay 1e-4 \
  --eval_strategy no \
  --save_steps 2000 \
  --save_total_limit 1 \
  --logging_steps 1 \
  --max_length 10240 \
  --warmup_ratio 0.03 \
  --dataloader_num_workers 4 \
  --eval_strategy no \
  --deepspeed zero1 \
  --system sysprompt.txt \
  --model /mnt/vlr/laishi/InternVL3-8B \
  --dataset /mnt/vlr/laishi/train_stage1_data.jsonl \
  --output_dir stage1_outputs \