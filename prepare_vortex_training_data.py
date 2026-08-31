"""
Prepare Vortex documentation for training.
Takes markdown files and converts them into training data the model can learn from.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

class VortexTrainingDataPrep:
    def __init__(self, docs_path: str):
        self.docs_path = docs_path
        self.conversations = []
        self.qa_pairs = []
        self.code_examples = []
        
    def extract_code_blocks(self, content: str) -> List[str]:
        """Extract all code blocks from markdown"""
        pattern = r'```(?:luau|lua)?\n(.*?)\n```'
        matches = re.findall(pattern, content, re.DOTALL)
        return matches
    
    def extract_sections(self, content: str) -> List[Tuple[str, str]]:
        """Extract header and content pairs"""
        sections = []
        parts = re.split(r'^(#{1,6}\s+.+?)$', content, flags=re.MULTILINE)
        
        for i in range(1, len(parts), 2):
            if i + 1 < len(parts):
                header = parts[i].replace('#', '').strip()
                body = parts[i + 1].strip()
                if header and body:
                    sections.append((header, body))
        return sections
    
    def create_qa_from_section(self, header: str, content: str) -> Dict:
        """Generate Q&A pairs from documentation sections"""
        lines = content.split('\n')
        text = ' '.join([line for line in lines if line.strip() and not line.startswith('```')])
        
        if len(text) < 20:
            return None

        questions = [
            f"What is {header.lower()}?",
            f"Explain {header.lower()} in Vortex",
            f"How do I use {header.lower()}?",
            f"Tell me about {header.lower()}",
        ]
        
        return {
            "questions": questions,
            "answer": text[:500],  # Limit to 500 chars
            "topic": header
        }
    
    def create_code_explanation_pairs(self, code: str, context: str) -> Dict:
        """Create question-answer pairs around code examples"""
        return {
            "question": f"Write Vortex/Luau code for: {context}",
            "answer": code,
            "type": "code_generation"
        }
    
    def process_markdown_file(self, filepath: str) -> None:
        """Process a single markdown file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Skip frontmatter
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    content = parts[2]
            
            sections = self.extract_sections(content)
            for header, body in sections:
                qa = self.create_qa_from_section(header, body)
                if qa:
                    self.qa_pairs.append(qa)
            
            codes = self.extract_code_blocks(content)
            for code in codes:
                if len(code.strip()) > 10:
                    self.code_examples.append({
                        "code": code.strip(),
                        "language": "luau",
                        "file": filepath
                    })
        
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    def process_all_docs(self) -> None:
        """Process all markdown files in the docs directory"""
        docs_dir = Path(self.docs_path)
        for md_file in docs_dir.rglob("*.md"):
            self.process_markdown_file(str(md_file))
        
        print(f"✓ Extracted {len(self.qa_pairs)} Q&A pairs")
        print(f"✓ Found {len(self.code_examples)} code examples")
    
    def create_conversation_pairs(self) -> List[Dict]:
        """Convert Q&A pairs to conversation format for fine-tuning"""
        conversations = []
        
        for qa_pair in self.qa_pairs:
            for question in qa_pair.get("questions", []):
                conversations.append({
                    "messages": [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": qa_pair.get("answer", "")}
                    ]
                })
        
        for code_ex in self.code_examples:
            conversations.append({
                "messages": [
                    {"role": "user", "content": f"Write Vortex code for: {code_ex.get('code', '')[:50]}..."},
                    {"role": "assistant", "content": code_ex.get("code", "")}
                ]
            })
        
        return conversations
    
    def save_jsonl(self, conversations: List[Dict], output_path: str) -> None:
        """Save conversations as JSONL (one JSON per line)"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for conv in conversations:
                f.write(json.dumps(conv) + '\n')
        print(f"✓ Saved {len(conversations)} conversations to {output_path}")
    
    def save_json(self, data: List[Dict], output_path: str) -> None:
        """Save data as JSON array"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved {len(data)} items to {output_path}")
    
    def create_system_prompts(self) -> List[Dict]:
        """Create specialized system prompts for different tasks"""
        return [
            {
                "task": "script_generation",
                "system": "You are an expert Vortex Luau programmer. Generate clean, well-commented Luau code for Vortex Studio. Include proper error handling and follow best practices."
            },
            {
                "task": "bug_fixing",
                "system": "You are an expert at debugging Vortex Luau scripts. Analyze code for bugs, security issues, and performance problems. Provide fixes with explanations."
            },
            {
                "task": "documentation",
                "system": "You are a Vortex Studio expert. Explain Vortex API concepts, classes, and methods clearly with code examples."
            },
            {
                "task": "world_generation",
                "system": "You are an expert at procedural generation in Vortex. Write Luau code to generate game worlds, structures, and environments."
            },
            {
                "task": "security_review",
                "system": "You are a security expert for Vortex games. Review Luau code for security vulnerabilities, exploits, and best practices."
            }
        ]
    
    def run_full_pipeline(self, output_dir: str = "vortex_training_data") -> None:
        """Run the whole pipeline from docs to training data."""
        os.makedirs(output_dir, exist_ok=True)

        print("\nPreparing Vortex training data")
        print("=" * 60)

        print("\n1. Reading documentation files...")
        self.process_all_docs()

        print("2. Creating conversation pairs...")
        conversations = self.create_conversation_pairs()

        print("3. Saving data...")
        self.save_jsonl(conversations, f"{output_dir}/vortex_conversations.jsonl")
        self.save_json(self.qa_pairs, f"{output_dir}/vortex_qa_pairs.json")
        self.save_json(self.code_examples, f"{output_dir}/vortex_code_examples.json")

        print("4. Setting up task prompts...")
        prompts = self.create_system_prompts()
        self.save_json(prompts, f"{output_dir}/system_prompts.json")

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Conversations: {len(conversations)}")
        print(f"Q&A pairs: {len(self.qa_pairs)}")
        print(f"Code examples: {len(self.code_examples)}")
        print(f"Saved to: {output_dir}")
        print("=" * 60)
        
        return output_dir

# Run preparation
if __name__ == "__main__":
    prep = VortexTrainingDataPrep("vortex-docs/content")
    prep.run_full_pipeline()
