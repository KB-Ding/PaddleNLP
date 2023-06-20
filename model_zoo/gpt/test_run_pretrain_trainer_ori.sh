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

CUDA_VISIBLE_DEVICES=1 /home/anaconda3/bin/python /home/PaddleNLP/model_zoo/gpt/run_pretrainer_trainer_ori.py \
    --model_type "gpt" \
    --model_name_or_path "gpt2-en" \
    --tokenizer_name_or_path "gpt2-en" \
    --input_dir "/home/PaddleNLP/model_zoo/ernie-1.0/preprocess/ori-data" \
    --output_dir "/home/PaddleNLP/model_zoo/ernie-1.0/preprocess/ori_gpt-output" \
    --split 949,50,1 \
    --max_seq_length 1024 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --fp16  \
    --fp16_opt_level "O2"  \
    --learning_rate 0.0001 \
    --min_learning_rate 0.00001 \
    --max_steps 10000 \
    --save_steps 5000 \
    --weight_decay 0.01 \
    --warmup_ratio 0.01 \
    --max_grad_norm 1.0 \
    --logging_steps 20 \
    --dataloader_num_workers 1 \
    --eval_steps 1000 \
    --report_to "visualdl" \
    --disable_tqdm true \
    --do_train \
    --do_eval \
    --device "gpu"