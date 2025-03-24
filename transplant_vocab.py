#!/usr/bin/env python3
"""
Vocab Transplantation Tool

All credit to turboderp for the original idea:

https://huggingface.co/turboderp/Qwama-0.5B-Instruct/blob/main/vocab_transplant.py
"""

import argparse
import json
import os
import shutil
import sys
from typing import Tuple, Dict

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig

def parse_arguments() -> argparse.Namespace:
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser(
        description="Transplant token embeddings between language models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("donor_dir", help="Path to donor model directory")
    parser.add_argument("target_dir", help="Path to target model directory")
    parser.add_argument("output_dir", help="Path to output model directory")
    parser.add_argument("--overwrite", action="store_true", 
                       help="Overwrite output directory if it exists")
    parser.add_argument("--unmapped-init-scale", type=float, default=0.0,
                       help="Scale factor [0-1] for initializing unmapped lm_head tokens")
    parser.add_argument("--use-cpu-only", action="store_true",
                       help="Use CPU only for model loading and processing in float32")
    parser.add_argument("--verbose", action="store_true",
                       help="Show detailed token mapping output")

    args = parser.parse_args()
    
    if not (0.0 <= args.unmapped_init_scale <= 1.0):
        sys.exit(f"Error: --unmapped-init-scale must be between 0.0 and 1.0 (got {args.unmapped_init_scale})")
        
    return args

def validate_directories(args: argparse.Namespace) -> None:
    """Validate input/output directory structure and permissions"""
    for dir_type, dir_path in [("donor", args.donor_dir), ("target", args.target_dir)]:
        if not os.path.isdir(dir_path):
            sys.exit(f"Error: {dir_type} directory does not exist: {dir_path}")
        if not os.access(dir_path, os.R_OK):
            sys.exit(f"Error: No read permissions for {dir_type} directory: {dir_path}")

    if os.path.exists(args.output_dir):
        if args.overwrite:
            if not os.access(args.output_dir, os.W_OK):
                sys.exit(f"Error: No write permissions for output directory: {args.output_dir}")
            shutil.rmtree(args.output_dir)
        else:
            sys.exit(f"Error: Output directory exists (use --overwrite to replace): {args.output_dir}")
    
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except OSError as e:
        sys.exit(f"Error: Failed to create output directory: {e}")

def load_model_config(path: str) -> dict:
    """Load model configuration"""
    config_path = os.path.join(path, "config.json")
    if not os.path.exists(config_path):
        sys.exit(f"Error: Config file not found at {config_path}")

    try:
        print(f"Loading config from '{path}'... ", end="")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        print("Done.")
    except Exception as e:
        sys.exit(f"Error loading config from {config_path}: {e}")

    return config

def load_model(path: str, torch_dtype=None) -> AutoModelForCausalLM:
    """Load model with error handling"""
    try:
        print(f"Loading model from '{path}'... ", end="")
        if torch_dtype is not None:
            model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", torch_dtype=torch_dtype)
        else:
            model = AutoModelForCausalLM.from_pretrained(path, device_map="cpu")  # Will load (and save) as torch.float32
        print("Done.")
        return model
    except Exception as e:
        sys.exit(f"Failed to load model: {e}")

def load_tokenizer(path: str) -> AutoTokenizer:
    """Load tokenizer with error handling"""
    try:
        print(f"Loading tokenizer from '{path}'... ", end="")
        tokenizer = AutoTokenizer.from_pretrained(path)
        print("Done.")
        return tokenizer
    except Exception as e:
        sys.exit(f"Failed to load tokenizer: {e}")

def main():
    args = parse_arguments()
    validate_directories(args)
    
    # Load tokenisers and configurations
    donor_config = load_model_config(args.donor_dir)
    donor_tokeniser = load_tokenizer(args.donor_dir)
    target_config = load_model_config(args.target_dir)
    target_tokeniser = load_tokenizer(args.target_dir)

    # Load the model
    if args.use_cpu_only:
        model = load_model(args.donor_dir)
    else:
        model = load_model(args.donor_dir, torch_dtype=donor_config.get("torch_dtype", None))

    # Validate vocab_size and hidden_size exists
        print(f"Loading config from '{source_model_path}'...")
    source_config = AutoConfig.from_pretrained(source_model_path, trust_remote_code=True)
    print("Done.")

    print(f"Loading config from '{target_model_path}'...")
    target_config = AutoConfig.from_pretrained(target_model_path, trust_remote_code=True)
    print("Done.")

    # Abrufen der Quell-Vokabulargröße (sowohl für flache als auch für verschachtelte Konfigurationen)
    if "text_config" in source_config and "vocab_size" in source_config["text_config"]:
        source_vocab_size = source_config["text_config"]["vocab_size"]
    else:
        assert "vocab_size" in source_config, "vocab_size not found in source model config"
        source_vocab_size = source_config["vocab_size"]
    print(f"Source vocab size: {source_vocab_size}")

    # Abrufen der Ziel-Vokabulargröße (sowohl für flache als auch für verschachtelte Konfigurationen)
    if "text_config" in target_config and "vocab_size" in target_config["text_config"]:
        target_vocab_size = target_config["text_config"]["vocab_size"]
    else:
        assert "vocab_size" in target_config, "vocab_size not found in target model config"
        target_vocab_size = target_config["vocab_size"]
    print(f"Target vocab size: {target_vocab_size}")


    print(f"Loading tokenizer from '{source_model_path}'...")
    source_tokenizer = AutoTokenizer.from_pretrained(source_model_path, trust_remote_code=True)
    print("Done.")
    print(f"Loading tokenizer from '{target_model_path}'...")
    target_tokenizer = AutoTokenizer.from_pretrained(target_model_path, trust_remote_code=True)
    print("Done.")

    # Initialize new embeddings
    donor_embed_tokens = model.model.embed_tokens.weight
    donor_lm_head = model.model.embed_tokens.weight if donor_config["tie_word_embeddings"] else model.lm_head.weight
    
    new_embed_tokens = torch.zeros(
        (target_vocab_size, donor_config["hidden_size"]),
        dtype=donor_embed_tokens.dtype,
        device=donor_embed_tokens.device
    )
    new_lm_head = torch.zeros(
        (target_vocab_size, donor_config["hidden_size"]),
        dtype=donor_lm_head.dtype,
        device=donor_lm_head.device
    )

    # Track mapping statistics
    one_to_one_count = 0
    two_to_one_count = 0
    three_plus_to_one_count = 0
    
    # Track lm_head statistics
    lm_head_set_count = 0
    lm_head_scaled_count = 0
    
    # Used to track already used prefix tokens for the lm_head
    used_prefix_tokens = set()
    
    # Configure progress display
    iterator = range(actual_vocab_size)
    if not args.verbose:
        iterator = tqdm(iterator, desc="Transplanting tokens", unit="token")
    
    for idx in iterator:
        decoded = target_tokeniser.decode([idx], decode_special_tokens=True)
        encoded = donor_tokeniser.encode(decoded, add_special_tokens=False, return_tensors="pt").flatten()

        if args.verbose:
            print(f"{idx:5d}: {repr(decoded)} → {encoded.tolist()}")

        # Track mapping types
        if encoded.numel() == 1:
            one_to_one_count += 1
        elif encoded.numel() == 2:
            two_to_one_count += 1
        else:
            three_plus_to_one_count += 1

        # Use only the final token of encoded sequence for input embeddings
        # NOTE: The rationale for doing this rather than turboderp's original idea to
        #       use the mean is because of the way the LLMs take the current token's
        #       embedding, perform lots of vector additions, and generate the next
        #       token - it's the transformer blocks that "look backward" so taking
        #       the mean doesn't really make sense... If the model is fine-tuned
        #       rather than used "as is"; then the different many-to-one embeddings
        #       should update the transformer blocks to look at the tokens that make
        #       up the start of the words (eg: "think-ing" and "drink-ing" will get
        #       assigned the same word embedding from our "last token only" method).
        new_embed_tokens[idx] = donor_embed_tokens[encoded[-1]]

        # Use only the first token for head embeddings (unless asked to use scaled mean)
        # NOTE: The rationale for doing this rather than turboderp's original idea to
        #       use the mean is because again, using the mean pulls in word endings
        #       that the model will only care about on the next token (eg: consider
        #       the "think-ing" and "drink-ing" example - the model will decide on the
        #       ending on the next auto-regressive pass; not on this pass).
        #       - The reason we are careful not to assign the same token more than once,
        #         is to not bias the probabilities using lots of different words using
        #         the same prefix stem (eg: "th-ink", "th-inking", "th-ought", etc).
        #       - The option of using a scaled down mean of the multi-token head
        #         embeddings is purely for further fine-tuning, as it will likely be a
        #         better initialisation than zeros to train from.
        #       - Mathematically, scaling the logits (via embeddings dot-product) makes 
        #         little sense, and to split tokens whilst keeping the rest of the
        #         probabilities correct we should *SUBTRACT* log(n) from each in the
        #         group of n-tokens we wish to split, but sadly there's no 'lm_head.bias'.
        #         (see Softmax's relation to "Independence of Irrelevant Alternatives").
        prefix_token = encoded[0].item()
        if prefix_token not in used_prefix_tokens:
            used_prefix_tokens.add(prefix_token)
            new_lm_head[idx] = donor_lm_head[prefix_token]
            lm_head_set_count += 1
        elif args.unmapped_init_scale > 0:
            encode_tokens = encoded.flatten()
            head_embeddings = donor_lm_head[encode_tokens]
            new_lm_head[idx] = head_embeddings.mean(dim=0) * args.unmapped_init_scale
            lm_head_scaled_count += 1

    # Print statistics
    print("\nTransplant mappings:")
    print(f"- 1 to 1  : {one_to_one_count} ({one_to_one_count/actual_vocab_size:.1%})")
    print(f"- 2 to 1  : {two_to_one_count} ({two_to_one_count/actual_vocab_size:.1%})")
    print(f"- 3+ to 1 : {three_plus_to_one_count} ({three_plus_to_one_count/actual_vocab_size:.1%})")
    
    print("\nHead initialized with:")
    lm_head_zeroed_count = target_vocab_size - (lm_head_set_count + lm_head_scaled_count)
    print(f"- Copies : {lm_head_set_count} ({lm_head_set_count/target_vocab_size:.1%})")
    if lm_head_scaled_count > 0:
        print(f"- Means  : {lm_head_scaled_count} ({lm_head_scaled_count/target_vocab_size:.1%})")
    print(f"- Zeros  : {lm_head_zeroed_count} ({lm_head_zeroed_count/target_vocab_size:.1%})")

    # Make a copy of the model's state_dict and get the type
    new_state_dict = model.state_dict().copy()
    old_dtype = model.model.embed_tokens.weight.dtype

    # Update the state_dict with new embeddings
    new_state_dict['model.embed_tokens.weight'] = new_embed_tokens.to(dtype=old_dtype)
    new_state_dict['lm_head.weight'] = new_lm_head.to(dtype=old_dtype)

    # Update model architecture
    model.model.embed_tokens.num_embeddings = target_vocab_size
    model.lm_head.out_features = target_vocab_size
    
    # Update model config
    model.config.update({
        'vocab_size': target_vocab_size,
        'bos_token_id': target_tokeniser.bos_token_id,
        'eos_token_id': target_tokeniser.eos_token_id,
    })
    
    # Add pad_token_id if it exists in the target tokeniser
    if target_tokeniser.pad_token_id is not None:
        model.config.update({'pad_token_id': target_tokeniser.pad_token_id})
    
    # Set tie_word_embeddings to False if it exists
    if hasattr(model.config, 'tie_word_embeddings'):
        model.config.update({'tie_word_embeddings': False})

    # Save final model and tokeniser
    print(f"\nSaving model and tokeniser to {args.output_dir}")
    model.save_pretrained(args.output_dir, state_dict=new_state_dict)
    target_tokeniser.save_pretrained(args.output_dir)
    
    print("Operation completed successfully")

if __name__ == "__main__":
    main()
