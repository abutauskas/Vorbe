"""
Fine-tune DeepSeek-Coder on Vortex documentation.
Makes the model understand Vortex-specific concepts better.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import json


class VortexCoderFineTuner:
    def __init__(self, model_name: str = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"):
        """Initialize the fine-tuner. Uses LoRA to keep training efficient."""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Device: {self.device}")
        print(f"Model: {model_name}")

        print("Loading tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        print("Setting up LoRA...")
        self.setup_lora()
    
    def setup_lora(self):
        """LoRA trains small adapter layers instead of every parameter - faster and less memory."""
        lora_config = LoraConfig(
            r=8,  # Dimension of the LoRA update
            lora_alpha=32,  # Scaling factor
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def load_training_data(self, jsonl_path: str) -> Dataset:
        """Load training data from JSONL - one JSON object per line."""
        print(f"Loading data from {jsonl_path}...")
        
        conversations = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                conversations.append(json.loads(line))
        
        def format_example(example):
            messages = example.get("messages", [])
            text = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                text += f"{role}: {content}\n"
            return {"text": text}
        
        data = [format_example(conv) for conv in conversations]
        dataset = Dataset.from_dict({
            "text": [d["text"] for d in data]
        })
        
        print(f"Loaded {len(dataset)} training examples")
        return dataset
    
    def tokenize_function(self, examples):
        """Tokenize, truncating to 2048 tokens to keep batches manageable."""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=2048,
            padding="max_length",
        )
    
    def fine_tune(self, 
                  jsonl_path: str,
                  output_dir: str = "vortex-coder-finetuned",
                  num_epochs: int = 3,
                  batch_size: int = 4,
                  learning_rate: float = 2e-4):
        print("\n" + "="*60)
        print("FINE-TUNING DEEPSEEK-CODER FOR VORTEX")
        print("="*60)
        
        dataset = self.load_training_data(jsonl_path)
        
        print("\nTokenizing...")
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=["text"]
        )
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            warmup_steps=100,
            save_steps=50,
            save_total_limit=3,
            logging_steps=10,
            optim="paged_adamw_8bit",
            fp16=self.device == "cuda",
            gradient_accumulation_steps=4,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
        )
        
        print("\nTraining...")
        trainer.train()

        print(f"\nSaving model to {output_dir}...")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print("\n" + "="*60)
        print("DONE!")
        print(f"Model saved to: {output_dir}")
        print("="*60)
        
        return output_dir
    
    @staticmethod
    def push_to_huggingface(model_dir: str, 
                            repo_name: str,
                            private: bool = False):
        """
        Push the fine-tuned model to Hugging Face so others can use it.
        """
        print(f"\nUploading to Hugging Face: {repo_name}...")
        
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)

        model.push_to_hub(
            repo_name,
            private=private,
            commit_message="Vorbe fine-tuned model"
        )
        tokenizer.push_to_hub(
            repo_name,
            private=private,
            commit_message="Vorbe fine-tuned tokenizer"
        )
        
        print(f"Model is up at: https://huggingface.co/{repo_name}")


if __name__ == "__main__":
    TRAINING_DATA_PATH = "vortex_training_data/vortex_conversations.jsonl"
    OUTPUT_DIR = "vortex-coder-deepseek-7b"
    HF_REPO = "abutauskas/vortex-coder-deepseek-7b"

    finetuner = VortexCoderFineTuner(
        model_name="deepseek-ai/deepseek-coder-7b-instruct-v1.5"
    )
    
    finetuner.fine_tune(
        jsonl_path=TRAINING_DATA_PATH,
        output_dir=OUTPUT_DIR,
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-4
    )
    
    # Uncomment to push to Hugging Face Hub
    # VortexCoderFineTuner.push_to_huggingface(OUTPUT_DIR, HF_REPO)
