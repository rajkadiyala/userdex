#!/usr/bin/env python3
"""Trains a Llama2 model on a corpora of documents.

Prior to execution, the caller must have agreed to the Meta Llama2 EULA:
https://ai.meta.com/resources/models-and-libraries/llama-downloads/.

Once approved, the caller must request access to the model on the HuggingFace
(HF) Hub using their HuggingFace account which should have the *same email* as
used on the Meta Llama2 EULA. Access to the Hub checkpoints can take ~1-2 business
days (in practice, this was turned around in ~30min-1hr). The caller should
obtain a user access token (UAT) from their HuggingFace account.

See more at:
* https://huggingface.co/docs/transformers/main/model_doc/llama2
* https://huggingface.co/docs/hub/security-tokens

Usage:
```shell
HF_UAT="$(cat /path/to/hf_uat)" && \
./train.py \
--hf_uat="${HF_UAT}" \
--model_name_or_path="meta-llama/Llama-2-7b-chat-hf" 2>&1 | tee /tmp/train_logs
```
"""

from collections.abc import Sequence

from absl import app
from absl import flags
from transformers import LlamaForCausalLM, LlamaTokenizer

_HF_UAT = flags.DEFINE_string(
    'hf_uat',
    None,
    help='A HuggingFace (HF) User Access Token (UAT) to access the HF Hub.')

_MODEL_NAME_OR_PATH = flags.DEFINE_string(
    'model_name_or_path',
    None,
    help=('The *model id* of a pretrained model hosted inside of a model repo '
          'on huggingface.co (see https://huggingface.co/meta-llama for a list '
          'of Llama2 models namespaced by meta-llama/), or a path to a '
          'directory containing model weights.'),
    required=True)


def main(args: Sequence[str]) -> None:
    del args  # Unused

    model = LlamaForCausalLM.from_pretrained(
        _MODEL_NAME_OR_PATH.value, token=_HF_UAT.value)
    tokenizer = LlamaTokenizer.from_pretrained(
        _MODEL_NAME_OR_PATH.value, token=_HF_UAT.value)

    # TODO(#5): Load data and train via MLM.
    # TODO(#6): Add support for LoRA.
    # TODO(#7): Add support for supervised-fine-tuning (SFT).


if __name__ == '__main__':
    app.run(main)
