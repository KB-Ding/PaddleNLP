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

from transformers import MT5ForConditionalGeneration, T5Tokenizer


def clean_up_codem_spaces(s: str):
    # post process
    # ===========================
    new_tokens = ["<pad>", "</s>", "<unk>", "\n", "\t", "<|space|>" * 4, "<|space|>" * 2, "<|space|>"]
    for tok in new_tokens:
        s = s.replace(f"{tok} ", tok)

    cleaned_tokens = ["<pad>", "</s>", "<unk>"]
    for tok in cleaned_tokens:
        s = s.replace(tok, "")
    s = s.replace("<|space|>", " ")
    # ===========================
    return s


def postprocess_text(preds):
    preds = [pred.strip() for pred in preds]
    return preds


def postprocess_code(preds):
    preds = [clean_up_codem_spaces(pred).strip() for pred in preds]
    return preds


model = MT5ForConditionalGeneration.from_pretrained("/home/models/pp-1024")
import pdb

pdb.set_trace()
tokenizer = T5Tokenizer.from_pretrained("/home/models/pp-1024")
article = "translate Japanese to Python: "
model_inputs = tokenizer(article, max_length=1024, return_tensors="pt")
# import pdb
# pdb.set_trace()
# batch = tokenizer.prepare_seq2seq_batch(src_texts=[article], return_tensors="pt")
gen_kwargs = {
    "max_length": 1024,
    "num_beams": 3,
    "length_penalty": 0,
    "min_length": 0,
}
output_ids = model.generate(
    input_ids=model_inputs.input_ids,
    **gen_kwargs,
)
# import pdb
# pdb.set_trace()
decoded_preds = [tokenizer.decode(output_ids[0])]
decoded_preds = postprocess_code(decoded_preds)
print(decoded_preds)
