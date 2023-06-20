# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

export DATA_DIR=/home/BookCorpus/books1/epubtxt
export PPNLP_HOME=/home/ppnlp
export HUGGINGFACE_HUB_CACHE=/home/hf_hub_cache
export HF_HOME=/home/hf_home
export TASK_NAME=SST-2

# python -u /home/PaddleNLP/examples/language_model/convbert/run_glue_trainer.py \
#     --model_type convbert \
#     --model_name_or_path convbert-small \
#     --task_name $TASK_NAME \
#     --max_seq_length 128 \
#     --batch_size 256   \
#     --learning_rate 1e-4 \
#     --num_train_epochs 3 \
#     --logging_steps 100 \
#     --save_steps 100 \
#     --output_dir /home/convbert_result/glue/$TASK_NAME/ \
#     --device gpu
unset CUDA_VISIBLE_DEVICES
python -m paddle.distributed.launch --gpus "3" /home/PaddleNLP/examples/language_model/convbert/run_glue_trainer.py \
    --model_name_or_path convbert-small \
    --task_name cola \
    --max_seq_length 128 \
    --per_device_train_batch_size 256   \
    --per_device_eval_batch_size 256   \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --logging_steps 50 \
    --save_steps 50 \
    --output_dir /home/convbert_result/glue \
    --device gpu \
    --fp16 False\
    --do_train \
    --do_eval
# python -u /home/PaddleNLP/examples/language_model/convbert/run_pretrain.py \
#     --model_type convbert \
#     --model_name_or_path /home/convbert_pretrain_model/model_250 \
#     --input_dir $DATA_DIR \
#     --output_dir /home/convbert_pretrain_model \
#     --train_batch_size 64 \
#     --learning_rate 5e-4 \
#     --max_seq_length 128 \
#     --weight_decay 1e-2 \
#     --adam_epsilon 1e-6 \
#     --warmup_steps 10000 \
#     --num_train_epochs 4 \
#     --logging_steps 50 \
#     --save_steps 50 \
#     --max_steps -1 \
#     --device gpu