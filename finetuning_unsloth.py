import argparse
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer

def main(args):
    # Set default tensor type for GPU optimization
    torch.set_default_tensor_type(torch.cuda.HalfTensor)

    # Load the pre-trained model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=args.dtype,
        load_in_4bit=args.load_in_4bit,
    )

    # Apply PEFT (Parameter-Efficient Fine-Tuning) to the model
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.random_state,
        use_rslora=args.use_rslora,
        loftq_config=None,
    )

    # Load and shuffle the dataset
    dataset = load_dataset("json", data_files=args.data_file, split="train")
    dataset = dataset.shuffle(seed=args.seed)

    # Initialize the trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=args.logging_steps,
            optim="adamw_8bit",
            weight_decay=args.weight_decay,
            lr_scheduler_type="linear",
            seed=args.seed,
            output_dir=args.output_dir,
        ),
    )
    

    # Train the model
    trainer.train()

    # Save the trained model
    model.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a language model with specified parameters.")
    parser.add_argument('--model_name', type=str, required=True, help="Name or path of the pre-trained model.")
    parser.add_argument('--max_seq_length', type=int, default=2048, help="Maximum sequence length.")
    parser.add_argument('--dtype', type=str, default=None, help="Data type for model parameters.")
    parser.add_argument('--load_in_4bit', action='store_true', help="Load model with 4-bit quantization.")
    parser.add_argument('--lora_r', type=int, default=64, help="LoRA rank parameter.")
    parser.add_argument('--lora_alpha', type=int, default=64, help="LoRA alpha parameter.")
    parser.add_argument('--lora_dropout', type=float, default=0.0, help="LoRA dropout rate.")
    parser.add_argument('--random_state', type=int, default=3407, help="Random seed for initialization.")
    parser.add_argument('--use_rslora', action='store_true', help="Use Rank Stabilized LoRA.")
    parser.add_argument('--data_file', type=str, required=True, help="Path to the training data file.")
    parser.add_argument('--batch_size', type=int, default=8, help="Training batch size per device.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="Number of gradient accumulation steps.")
    parser.add_argument('--warmup_steps', type=int, default=3, help="Number of warmup steps.")
    parser.add_argument('--num_train_epochs', type=int, default=1, help="Number of training epochs.")
    parser.add_argument('--learning_rate', type=float, default=2e-4, help="Learning rate.")
    parser.add_argument('--logging_steps', type=int, default=1, help="Logging interval in steps.")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay rate.")
    parser.add_argument('--seed', type=int, default=3407, help="Random seed.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the trained model.")
    args = parser.parse_args()

    main(args)
