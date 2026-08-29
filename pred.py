import os
import time
from datasets import load_dataset
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.distributed as dist
from ecm.method.ecm import enable_ecm_attention, ECMCache


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Llama-3.1-8B-Instruct",
                        choices=["Mistral-7B-Instruct-v0.3", "Llama-3.1-8B-Instruct",
                                 "Qwen2-7B-Instruct"])
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--ratio', type=float,
                        help="Compression ratio, length * ratio = recent_size, "
                             "fix to 2048 if the param is not set.")
    parser.add_argument('--start_size', type=int, default=32,
                        help="Start window size, which will not be compressed.")
    return parser.parse_args(args)


def build_chat(prompt, model_name, tokenizer):
    if "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "Llama-3" in model_name:
        prompt = (f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                  f"{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")
    elif "Qwen" in model_name or "Mistral" in model_name:
        chat = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return prompt


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, device):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        path, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0
    model = model.eval()
    return model, tokenizer


@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len,
                    model_name="llama", dataset=""):
    generated_ids = [input_ids.item()]
    pred_token_idx = input_ids
    is_code = dataset in ["lcc", "repobench-p"]
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        generated_ids.append(pred_token_idx.item())
        generated_text = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=not is_code,
            spaces_between_special_tokens=False,
        )
        if "llama" in model_name.lower() and (
            pred_token_idx[0].item() == tokenizer.eos_token_id
            or pred_token_idx[0].item() == 128009
        ):
            break
        if "qwen" in model_name.lower() and (
            pred_token_idx[0].item() == tokenizer.eos_token_id
            or tokenizer.decode(pred_token_idx[0]) == "<|im_end|>"
        ):
            break
        if "mistral" in model_name.lower() and (
            pred_token_idx[0].item() == tokenizer.eos_token_id
            or tokenizer.decode(pred_token_idx[0]) == "[/INST]"
        ):
            break
    return generated_text


def get_pred(model, tokenizer, rank, world_size, data_all, max_gen, prompt_format,
             dataset, device, model_name, out_path, args):
    data = data_all[rank::world_size]
    k_seq_dim = v_seq_dim = 2

    enable_ecm_attention(model_name, model)

    for json_obj in tqdm(data, desc=f"Processing dataset {dataset} on rank {rank}"):
        start_size = args.start_size
        if args.ratio is not None:
            assert 0 < args.ratio <= 1, "The ratio should be in (0, 1]"
            recent_size = int(args.ratio * json_obj["length"])
        else:
            recent_size = 2048

        kv_cache = ECMCache(
            start_size=start_size,
            recent_size=recent_size,
            k_seq_dim=k_seq_dim,
            v_seq_dim=v_seq_dim,
        )

        past_key_values = None
        if dataset in ["lsht", "trec", "triviaqa", "samsum"] or dataset in ["lcc", "repobench-p"]:
            prompt = prompt_format.format(**json_obj)
        else:
            prompt = build_chat(prompt_format.format(**json_obj), model_name, tokenizer)

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = input.input_ids.shape[-1]

        input_window = 512

        # Timing / memory measurement
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start_time = time.perf_counter()

        for idx in range(0, context_length - 1, input_window):
            if idx + input_window < context_length:
                input_ids = input.input_ids[:, idx: idx + input_window].to(device)
            else:
                input_ids = input.input_ids[:, idx:].to(device)

            if kv_cache is not None and past_key_values is not None:
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids,
                        past_key_values=past_key_values,
                        use_cache=True,
                        output_attentions=True,
                    )
                attns = outputs.attentions
                attns = [attn[:, :, :, :-input_ids.shape[-1]] for attn in attns]
                past_key_values = kv_cache(past_key_values, attns)

            with torch.no_grad():
                outputs = model(
                    input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = outputs.past_key_values

        torch.cuda.synchronize(device)
        end_time = time.perf_counter()
        time_cost = end_time - start_time
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        memory_usage = peak_memory_bytes / (1024 * 1024)

        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        pred = greedy_generate(model, tokenizer, pred_token_idx, past_key_values,
                               max_gen_len=max_gen, model_name=model_name, dataset=dataset)

        with open(out_path, "a", encoding="utf-8") as f:
            json.dump({
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj.get("all_classes", []),
                "length": json_obj["length"],
                "recent_size": recent_size,
                "start_size": start_size,
                "time_cost": time_cost,
                "memory_usage": memory_usage,
            }, f, ensure_ascii=False)
            f.write('\n')

    if world_size > 1:
        dist.barrier()


if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    # NOTE: This prediction pipeline has only been validated in single-process execution.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        dist.init_process_group(backend='nccl')

    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model

    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report",
                    "multi_news", "trec", "triviaqa", "samsum", "passage_count",
                    "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["2wikimqa"]

    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))

    if local_rank == 0:
        if not os.path.exists("pred"):
            os.makedirs("pred")
        if not os.path.exists("pred_e"):
            os.makedirs("pred_e")

    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    model, tokenizer = load_model_and_tokenizer(model2path[model_name], device)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sub_dir = "pred_e" if args.e else "pred"

    out_dir = f"{sub_dir}/{model_name}/ecm_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    for dataset in datasets:
        suffix = "_e" if args.e else ""
        local_jsonl = os.path.join(
            os.environ.get("LONGBENCH_DATA_DIR", "./data"),
            f"{dataset}{suffix}.jsonl"
        )
        data = load_dataset("json", data_files={"test": local_jsonl}, split="test")

        out_path = f"{out_dir}/{dataset}.jsonl"

        if local_rank == 0 and os.path.exists(out_path):
            os.remove(out_path)

        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        data_all = [data_sample for data_sample in data]

        get_pred(model, tokenizer, local_rank, world_size, data_all, max_gen, prompt_format,
                 dataset, device, model_name, out_path, args)

    if world_size > 1:
        dist.destroy_process_group()

    print(f"Pred has been saved to {out_dir}")
