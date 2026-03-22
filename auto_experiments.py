import argparse
import os
import json
import subprocess
import time
# This script automates a full pipeline of experiments combining different artists and poison methods.

RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiments_names",
        type=str,
        nargs='+',
        help="One or more experiment names to load and run sequentially.",
    )
    parser.add_argument(
        "--set_index",
        type=int,
        nargs=2,
        default=[-1, -1],
        help="Two experiment progress indices for initialization (default: [-1, -1])",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="If set, will only print the generated commands without executing them.",
    )
    
    args = parser.parse_args()
    return args


def build_experiments_cmds(experiments_name):
    backbone_base_model_path = "/root/autodl-fs/stable-diffusion-model"
    benchmark_base_model_path = "/root/autodl-fs/stable-diffusion-model"
    clip_dir = "/root/autodl-tmp/clip-vit-base-patch32"
    result_root = "/root/autodl-tmp/data"
    artist_data_root = "/root/autodl-tmp/data/artist_data"
    artists = [
        'bea-ticex','jaden-wan','junjiuk','mark-komorowski','tanner-staheli','Claude_Monet',]
        # 'Gustav_Klimt','Pablo_Picasso','Salvador_Dalí','Vincent_van_Gogh', 'Käthe_Kollwitz', 'Albrecht_Durer']
    
    experiments = []
    for artist in artists:
        artist_dir = f"{artist_data_root}/{artist}"
        poison_data_dir = f"{artist_dir}/poison"
        
        poison_dreambooth_dir = f"{result_root}/poisoned_model/dreambooth"
        poisoned_model_dir = f"{result_root}/poisoned_model/full_finetune"
        textual_inversion_dir = f"{result_root}/poisoned_model/textual_inversion"
        
        ff_generated_poison_img_dir = f"{result_root}/generated_images/full_finetune/{artist}/{experiments_name}"
        dl_generated_poison_img_dir = f"{result_root}/generated_images/dreambooth_lora/{artist}/{experiments_name}"
        ti_generated_poison_img_dir = f"{result_root}/generated_images/textual_inversion/{artist}/{experiments_name}"
        
        os.makedirs(poisoned_model_dir, exist_ok=True)
        os.makedirs(poison_dreambooth_dir, exist_ok=True)
        os.makedirs(textual_inversion_dir, exist_ok=True)
        
        os.makedirs(ff_generated_poison_img_dir, exist_ok=True)
        os.makedirs(dl_generated_poison_img_dir, exist_ok=True)
        os.makedirs(ti_generated_poison_img_dir, exist_ok=True)

        experiments.append({
            
        "cmds":{
            "poison_attack":f"""
            python3 poisoning.py --pretrained_model="{backbone_base_model_path}" \\
                --artist_dir="{artist_dir}" \\
                --block_size=32 """,
            
#             "gaussian_defense": f"""
#             python3 defense.py --input_folder="{poison_data_dir}" --output_folder="{poison_data_dir}" --method Gaussian --param 8 """,

#             "full_finetune_poisoned_model":f"""
#             accelerate launch train_text_to_image.py --pretrained_model_name_or_path="{benchmark_base_model_path}" \\
#                 --train_data_dir="{poison_data_dir}" \\
#                 --use_ema --center_crop --random_flip --gradient_checkpointing \\
#                 --enable_xformers_memory_efficient_attention --lr_scheduler="constant" \\
#                 --resolution=512 --seed=123456 --train_batch_size=1 \\
#                 --gradient_accumulation_steps=1 --mixed_precision="fp16" \\
#                 --max_train_steps=1600 --checkpointing_steps=5000 \\
#                 --learning_rate=5e-6 --max_grad_norm=1 --lr_warmup_steps=0 \\
#                 --output_dir="{poisoned_model_dir}" """,

#             "full_finetune_generate_poisoned_images":f"""
#             python3 generate.py --test_metadata_path="{artist_dir}/test/metadata.jsonl" \\
#                 --model_dir="{poisoned_model_dir}" --output_dir="{ff_generated_poison_img_dir}" --model_type="full" """,
                
#             "full_finetune_evaluate":f"""
#             python3 evaluation.py --clip_dir="{clip_dir}" \\
#                 --generated_dir="{result_root}/generated_images" \\
#                 --output_dir="{result_root}/result/full_finetune" \\
#                 --experments_name="{experiments_name}" \\
#                 --finetuning_method="full_finetune" """,
            
#             "dreambooth_lora_poisoned_model": f"""
#             bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate dreambooth39 && accelerate launch train_dreambooth_lora.py \\
#                 --pretrained_model_name_or_path="{benchmark_base_model_path}" \\
#                 --instance_data_dir="{poison_data_dir}" \\
#                 --output_dir="{poison_dreambooth_dir}" \\
#                 --instance_prompt="a painting by {artist}" \\
#                 --resolution=512 \\
#                 --train_batch_size=1 \\
#                 --gradient_accumulation_steps=1 \\
#                 --checkpointing_steps=2000 \\
#                 --learning_rate=1e-4 \\
#                 --lr_scheduler="constant" \\
#                 --lr_warmup_steps=0 \\
#                 --max_train_steps=1600 \\
#                 --validation_prompt="a painting by {artist}" \\
#                 --validation_epochs=50 \\
#                 --seed=123456' """,

#             "dreambooth_lora_generate_poisoned_images":f"""
#             bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate dreambooth39 && accelerate launch generate.py \\
#                 --test_metadata_path="{artist_dir}/test/metadata.jsonl" \\
#                 --pre_model_dir="{benchmark_base_model_path}" --model_dir="{poison_dreambooth_dir}" --output_dir="{dl_generated_poison_img_dir}" --model_type="dreambooth"' """,

#             "dreambooth_lora_evaluate":f"""
#             python3 evaluation.py --clip_dir="{clip_dir}" \\
#                 --generated_dir="{result_root}/generated_images" \\
#                 --output_dir="{result_root}/result/dreambooth_lora" \\
#                 --experments_name="{experiments_name}" \\
#                 --finetuning_method="dreambooth_lora" """,
                
#             "textual_inversion":f"""
#             bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate dreambooth39 && accelerate launch textual_inversion.py \\
#                 --pretrained_model_name_or_path="{benchmark_base_model_path}" \\
#                 --train_data_dir="{poison_data_dir}" \\
#                 --learnable_property="style" \\
#                 --placeholder_token="<{artist}>" \\
#                 --initializer_token="style" \\
#                 --resolution=512 \\
#                 --train_batch_size=1 \\
#                 --gradient_accumulation_steps=1 \\
#                 --max_train_steps=2000 \\
#                 --learning_rate=5.0e-04 \\
#                 --scale_lr \\
#                 --lr_scheduler="constant" \\
#                 --lr_warmup_steps=0 \\
#                 --seed=123456 \\
#                 --output_dir="{textual_inversion_dir}" '""",
                
#             "textual_inversion_generate_poisoned_images":f"""
#             bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate dreambooth39 && accelerate launch generate.py \\
#                 --test_metadata_path="{artist_dir}/test/metadata.jsonl" \\
#                 --pre_model_dir="{benchmark_base_model_path}" --model_dir="{textual_inversion_dir}" --output_dir="{ti_generated_poison_img_dir}" --model_type="textual_inversion"' """,

#             "textual_inversion_evaluate":f"""
#             python3 evaluation.py --clip_dir="{clip_dir}" \\
#                 --generated_dir="{result_root}/generated_images" \\
#                 --output_dir="{result_root}/result/textual_inversion" \\
#                 --experments_name="{experiments_name}" \\
#                 --finetuning_method="textual_inversion" """,
        },
            
        "artist":f"{artist}",
        "experiments_name":f"{experiments_name}",
            
        })
            
    return experiments
    
    
def save_experiment_details(experiment_path, commands, current_index):
    # Saving commands and current index (now as a dictionary)
    experiment_details = {
        "commands": commands,
        "current_task": current_index[0],
        "current_step": current_index[1],
    }

    with open(experiment_path, "w") as f:
        json.dump(experiment_details, f, indent=4)

    print(f"{CYAN}[INFO] Saved progress at task {current_index[0]}, step {current_index[1]}.")
    
    
def load_experiment_details(experiment_path):
    if not os.path.exists(experiment_path):
        raise FileNotFoundError(f"Experiment file not found: {experiment_path}")
    
    with open(experiment_path, "r") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "commands" not in data or "current_task" not in data or "current_step" not in data:
        raise ValueError("Invalid experiment file format. Must be a dictionary with 'commands' and 'current_index'.")
    
    commands = data["commands"]
    current_index = [data["current_task"],data["current_step"]]
    
    print(f"{CYAN}[INFO] Loaded {len(commands)} experiment steps. Resuming from task {current_index[0]}, step {current_index[1]}.")

    return commands, current_index


def main(args, experiments_path, experiments_name):

    # Initialize experimental information
    if os.path.exists(experiments_path):
        print(f"{CYAN}[INFO] A previous experiment record was found. {RESET}")

        if args.set_index == [-1,-1]:
            commands, current_index = load_experiment_details(experiments_path)
            print(f"{CYAN}[INFO] Resuming experiment from task {current_index[0] + 1}/{len(commands)}{RESET}, step {current_index[1] + 1}")
        else:
            commands, _ = load_experiment_details(experiments_path)
            current_index = args.set_index
            print(f"{CYAN}[INFO] Resuming experiment from task {current_index[0] + 1}/{len(commands)}{RESET}, step {current_index[1] + 1}")

    else:
        print(f"{CYAN}[INFO] No existing experiment found. Creating new one.{RESET}")
        commands = build_experiments_cmds(experiments_name)
        if args.set_index == [-1,-1]:
            current_index = [0,0]
        else:
            current_index = args.set_index
        
    save_experiment_details(experiments_path, commands, current_index)

    # Execute command blocks one by one
    for i in range(current_index[0], len(commands)):
        command_block = commands[i]
        artist = command_block.get("artist", "Unknown")
        experiments_name = command_block.get("experiments_name", "Unknown")
        command_block = command_block['cmds']
        
        print(f"\n{YELLOW}[INFO] ===== Executing Command Group {i + 1}/{len(commands)} | Steps: {len(command_block)} steps | Artist: {artist} | experiments: {experiments_name} ====={RESET}\n")

        for step_idx, (step_name, cmd) in enumerate(command_block.items()):
            # skip the step that had completed
            if step_idx < current_index[1]:
                continue

            print(f"{BLUE}[{time.strftime('%Y-%m-%d %H:%M:%S')}] [STEP {step_idx + 1}/{len(command_block)}] >> {step_name}{RESET}")
            print("-" * 80)
            print(f"{GREEN}{cmd.strip()}{RESET}")
            print("-" * 80)
            
            if not args.dry_run:
                try:
                    subprocess.run(cmd.strip(), shell=True, check=True)
                    print(f"{GREEN}[INFO] ✅ Step '{step_name}' completed.{RESET}\n")
                except subprocess.CalledProcessError as e:
                    print(f"\n{RED}[ERROR] ❌ Step '{step_name}' in group {i + 1} failed. Halting execution.{RESET}")
                    print(f"        Error Message: {e}\n", flush=True)
                    exit(1)

            # save step
            current_index[1] = step_idx + 1
            save_experiment_details(experiments_path, commands, current_index)

        # save task
        current_index[0] = i + 1
        current_index[1] = 0
        save_experiment_details(experiments_path, commands, current_index)
        print(f"{GREEN}[INFO] ✅ Finished command group {i + 1}/{len(commands)}. Progress saved.{RESET}\n")

    print(f"{CYAN}[INFO] 🎉 All experiment steps completed successfully!{RESET}")
    
    
if __name__ == "__main__":
    args = parse_args()
    experiments_base_path = "/root/autodl-tmp/experiments/"

    # 确保 experiments 目录存在
    os.makedirs(experiments_base_path, exist_ok=True)

    for exp_name in args.experiments_names:
        # 拼接当前实验的路径
        experiments_path = os.path.join(experiments_base_path, f"{exp_name}.json")

        print(f"\n{YELLOW}===== Running experiment: {exp_name} ====={RESET}")
        print(f"{YELLOW}Saving/loading experiment record at: {experiments_path}{RESET}")

        # 运行 main()，传入当前 experiment 的 path
        main(args=args, experiments_path=experiments_path, experiments_name = exp_name)