"""
Hugging Face Spaces Deployment
Deploy Vortex-trained model as a web interface
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

class VorbeAssistant:
    def __init__(self, model_name: str = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"):
        """Initialize the assistant with the fine-tuned model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.model.eval()

        try:
            with open("vortex_training_data/system_prompts.json", 'r') as f:
                self.system_prompts = {p['task']: p['system'] for p in json.load(f)}
        except:
            self.system_prompts = {}
    
    def generate_code(self, prompt: str, task_type: str = "script_generation", max_tokens: int = 512):
        """Generate Vortex Luau code"""
        system = self.system_prompts.get(task_type, "You are an expert Vortex Luau programmer.")
        
        full_prompt = f"{system}\n\nUser: {prompt}\n\nAssistant:"
        
        inputs = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=len(inputs[0]) + max_tokens,
                temperature=0.7,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the assistant's response
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        return response
    
    def chat(self, message: str, task_type: str, history: list):
        """Chat interface"""
        response = self.generate_code(message, task_type)
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        return response, history


def create_gradio_interface():
    """Create Gradio web interface"""
    assistant = VorbeAssistant()

    task_options = [
        ("Script Generation", "script_generation"),
        ("Bug Fixing", "bug_fixing"),
        ("Documentation", "documentation"),
        ("World Generation", "world_generation"),
        ("Security Review", "security_review"),
    ]
    
    with gr.Blocks(title="Vorbe", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# Vorbe")
        gr.Markdown("AI-powered assistant for Vortex Studio scripting. Choose a task and describe what you need.")
        
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Chat", height=400)
                
            with gr.Column(scale=1):
                task_type = gr.Dropdown(
                    choices=[t[0] for t in task_options],
                    value="Script Generation",
                    label="Task Type",
                    info="Select what you need help with"
                )
                
                examples = gr.Examples(
                    examples=[
                        "Write a script that spawns a red part at (0, 10, 0)",
                        "Debug this code: local part = workspace.Part; part.Color = Color3.new(1, 0, 0)",
                        "Explain how to use RemoteEvents in Vortex",
                        "Generate a procedural dungeon using Vortex",
                        "Review this code for security issues: player:Kick()",
                    ],
                    inputs=None,
                    label="Example Prompts"
                )
        
        msg = gr.Textbox(
            label="Your Message",
            placeholder="Describe what you want to create or fix...",
            lines=2
        )
        
        with gr.Row():
            submit_btn = gr.Button("Send", variant="primary")
            clear_btn = gr.Button("Clear Chat")
        
        history_state = gr.State(value=[])
        
        def task_to_type(task_name: str) -> str:
            """Convert task name to type"""
            for name, type_val in task_options:
                if name == task_name:
                    return type_val
            return "script_generation"
        
        def submit(message, task_name, history):
            if not message:
                return chatbot, history_state
            
            task_type_val = task_to_type(task_name)
            response, history = assistant.chat(message, task_type_val, history)
            
            return history, history
        
        submit_btn.click(
            fn=submit,
            inputs=[msg, task_type, history_state],
            outputs=[chatbot, history_state]
        ).then(
            lambda: ("", []),
            outputs=[msg, history_state]
        )
        
        msg.submit(
            fn=submit,
            inputs=[msg, task_type, history_state],
            outputs=[chatbot, history_state]
        ).then(
            lambda: ("", []),
            outputs=[msg, history_state]
        )
        
        clear_btn.click(
            lambda: ([], []),
            outputs=[chatbot, history_state]
        )
        
        gr.Markdown("""
        ## How to use

        1. Pick a task type: Script Generation, Bug Fixing, and so on.
        2. Describe what you need.
        3. The assistant replies with code, an explanation, or a review.

        ## Features

        - Script Generation: create Vortex Luau scripts from descriptions
        - Bug Fixing: get help debugging and fixing code issues
        - Documentation: learn about Vortex APIs and concepts
        - World Generation: generate procedural game worlds
        - Security Review: analyze code for security vulnerabilities

        ## Tips

        - Be specific in your descriptions
        - Include code snippets if you need help debugging
        - Ask follow-up questions for clarification
        """)
    
    return interface


if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
