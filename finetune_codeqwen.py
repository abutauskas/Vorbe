"""
Fine-tune CodeQwen on Vortex Documentation
Alibaba's powerful code-focused model
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
from peft import LoraConfig, get_peft_model
import json

class VortexCodeQwenFineTuner:
    def __init__(self, model_name: str = "Qwen/CodeQwen1.5-7B"):
        """Initialize CodeQwen fine-tuner"""
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎯 Using device: {self.device}")
        print(f"📦 Model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.setup_lora()
    
    def setup_lora(self):
        """Setup LoRA for efficient training"""
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.model = get_peft_model(self.model, lora_config)
    
    def load_training_data(self, jsonl_path: str) -> Dataset:
        """Load training data"""
        conversations = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                conversations.append(json.loads(line))
        
        texts = []
        for conv in conversations:
            messages = conv.get("messages", [])
            text = ""
            for msg in messages:
                text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            texts.append(text)
        
        dataset = Dataset.from_dict({"text": texts})
        return dataset
    
    def fine_tune(self, jsonl_path: str, output_dir: str = "vortex-codeqwen-finetuned"):
        """Fine-tune CodeQwen"""
        print("\n" + "="*60)
        print("🚀 FINE-TUNING CODEQWEN FOR VORTEX")
        print("="*60)
        
        dataset = self.load_training_data(jsonl_path)
        
        tokenized = dataset.map(
            lambda x: self.tokenizer(x["text"], truncation=True, max_length=2048, padding="max_length"),
            batched=True,
            remove_columns=["text"]
        )
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,
            learning_rate=2e-4,
            weight_decay=0.01,
            warmup_steps=100,
            save_steps=100,
            logging_steps=10,
            fp16=self.device == "cuda",
            gradient_accumulation_steps=4,
            optim="paged_adamw_8bit",
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized,
            tokenizer=self.tokenizer,
        )
        
        trainer.train()
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"\n✅ Model saved to {output_dir}")
        return output_dir


if __name__ == "__main__":
    finetuner = VortexCodeQwenFineTuner()
    finetuner.fine_tune("vortex_training_data/vortex_conversations.jsonl")
