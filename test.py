import numpy as np
import torch
from llm_attacks.minimal_gcg.opt_utils import load_model_and_tokenizer
import argparse
from utils.utils import load_model_and_tokenizer
from utils.string_utils import load_prompts, save_test_to_file
from attacks import  GCG

def load_defense_model_path(args):
    if "llama-2-7b" in args.base_model:
        args.model_path = "../llm_models/llama-2-7b-chat-hf"
    else:
        raise  NameError
###########################################################
def defense_adv_prompts(goals, targets, model,tokenizer, device, args):
    adv_instructions = []
    i = args.start_index
    for goal_i, target_i in zip(goals[i:], targets[i:]):

        pert_goal_i = goal_i
        print(f"""\n{'=' * 36}\nAttack Method: {args.attack}\n{'=' * 36}\n""")
        if args.attack == "GCG":
            adv_prompt, model_output, iteration, is_JB = GCG(model=model, tokenizer=tokenizer, device=device, goal=pert_goal_i, target=target_i,args= args)
        else:
            raise NameError


        adv_instructions.append({
            "original_prompt": goal_i,
            "per_prompt": pert_goal_i,
            "target": target_i,
            "adv_prompt": adv_prompt,
            "language_model_output": model_output,
            "attack_iterations": iteration,
            "data_id": i,
            "is_JB": is_JB
        })
        save_test_to_file(args = args, instructions = adv_instructions)
        i += 1
    return adv_instructions

def main(args):
    np.random.seed(20)
    # Set the random seed for PyTorch
    torch.manual_seed(20)
    # If you are using CUDA (i.e., a GPU), also set the seed for it
    torch.cuda.manual_seed_all(20)

    # default setting
    # load_defense_model_path(args)
    model_path = args.model_path
    print("\n\nmodel path",model_path,"\n\n")
    device = 'cuda:{}'.format(args.device_id)
    instructions_path = args.instructions_path

    goals, targets = load_prompts(instructions_path)

    model, tokenizer = load_model_and_tokenizer(model_path, tokenizer_path=None, device=device)
    defense_adv_prompts(goals, targets, model, tokenizer, device, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ########### Attack model parameters ##########
    parser.add_argument(
        "--base_model",
        default = "Meta-Llama-3-8B",
        help = "Name of base model for target model",
        choices=["vicuna", "llama-2", "gpt-3.5-turbo", "gpt-4", "claude-instant-1","claude-2", "palm-2","guard-llama",
                 "Meta-Llama-3-8B-Instruct"]
    )
    parser.add_argument(
        "--model_path",
        type = str,
        default = "../llm_models/llama-2-7b-chat-hf",
        help = "The model path of target model",
    )
    parser.add_argument(
        "--template_name",
        type = str,
        default = "llama-2",
        help = "Template name of LLMs",
    )
    parser.add_argument(
        "--check_type",
        type = str,
        default = "prefixes",
        choices= ["agent", "prefixes"],
        help = "The check type for jailbreak content."
    )
    parser.add_argument(
        "--instructions_path",
        type = str,
        default = "data/adv_harmful_bench/harmful_bench_test.csv",
        help = "The path of instructions"
    )
    parser.add_argument(
        "--save_path_adv_prompt",
        type = str,
        default = "exp_results/test/adv_prompts/",
        help = "The model path of large language model"
    )
    parser.add_argument(
        "--file_name_adv_prompt",
        type = str,
        default = "_8epoch",
        help = "The model path of large language model"
    )
    parser.add_argument(
        "--device_id",
        type = int,
        default = '0',
        help = "device id"
    )
    ##################Attack Methods################################
    parser.add_argument(
        "--attack",
        type = str,
        default = "GCG",
        help = "Name of attack method",
        choices=["GCG"]
    )
    parser.add_argument(
        "--start_index",
        type = int,
        default = 0,
        help = "Start index"
    )
    args = parser.parse_args()
    main(args)


