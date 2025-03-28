import os
import json
import time
from typing import Dict, List, Optional, Tuple
from enum import Enum
import tqdm
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress
import requests
from cryptography.fernet import Fernet
import tiktoken
from pydantic import BaseModel, validator

# Constants
CONFIG_FILE = "llm_config.json"
API_KEYS_FILE = "api_keys.enc"
MODELS = {
    "siliconflow": {
        "base_url": "https://api.siliconflow.ai/v1",
        "token_encoder": "cl100k_base"
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "token_encoder": "cl100k_base"
    },
    "groq": {
        "base_url": "https://api.groq.com/v1",
        "token_encoder": "cl100k_base"
    }
}

class ModelSettings(BaseModel):
    name: str
    api_url: str
    model_id: str
    temperature: float = 0.7
    top_p: float = 1.0
    context: str = ""
    role: str = "assistant"
    input_cost: float
    output_cost: float
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not 0 <= v <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v
    
    @validator('top_p')
    def validate_top_p(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Top P must be between 0 and 1")
        return v

class LLMClient:
    def __init__(self):
        self.console = Console()
        self.encryption_key = self._get_encryption_key()
        self.config = self._load_config()
        self.api_keys = self._load_api_keys()
        self.conversations = {}
        self.current_conversation = None
        
    def _get_encryption_key(self) -> bytes:
        if not os.path.exists("encryption.key"):
            key = Fernet.generate_key()
            with open("encryption.key", "wb") as f:
                f.write(key)
        else:
            with open("encryption.key", "rb") as f:
                key = f.read()
        return key
    
    def _load_config(self) -> Dict:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        return {"models": [], "default_model": None}
    
    def _load_api_keys(self) -> Dict:
        cipher_suite = Fernet(self.encryption_key)
        if os.path.exists(API_KEYS_FILE):
            with open(API_KEYS_FILE, "rb") as f:
                encrypted = f.read()
                decrypted = cipher_suite.decrypt(encrypted)
                return json.loads(decrypted.decode())
        return {}
    
    def _save_api_keys(self):
        cipher_suite = Fernet(self.encryption_key)
        encrypted = cipher_suite.encrypt(json.dumps(self.api_keys).encode())
        with open(API_KEYS_FILE, "wb") as f:
            f.write(encrypted)
    
    def _save_config(self):
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.config, f, indent=2)
    
    def _get_tokenizer(self, model_type: str) -> tiktoken.Encoding:
        try:
            return tiktoken.get_encoding(MODELS[model_type]["token_encoder"])
        except KeyError:
            return tiktoken.get_encoding("cl100k_base")
    
    def calculate_tokens(self, text: str, model_type: str) -> int:
        tokenizer = self._get_tokenizer(model_type)
        return len(tokenizer.encode(text))
    
    def estimate_cost(self, input_tokens: int, output_tokens: int, model_settings: ModelSettings) -> float:
        input_cost = (input_tokens / 1_000_000) * model_settings.input_cost
        output_cost = (output_tokens / 1_000_000) * model_settings.output_cost
        return input_cost + output_cost
    
    def verify_model(self, model_type: str, api_key: str) -> bool:
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            response = requests.get(
                f"{MODELS[model_type]['base_url']}/models",
                headers=headers,
                timeout=10
            )
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def add_model(self, model_settings: ModelSettings, api_key: str):
        if model_settings.name in [m["name"] for m in self.config["models"]]:
            raise ValueError("Model with this name already exists")
        
        model_type = next(
            (k for k, v in MODELS.items() if v["base_url"] in model_settings.api_url),
            None
        )
        if not model_type:
            raise ValueError("Unsupported model API URL")
        
        if not self.verify_model(model_type, api_key):
            raise ValueError("Failed to verify model with provided API key")
        
        self.api_keys[model_settings.name] = api_key
        self.config["models"].append(model_settings.dict())
        if not self.config["default_model"]:
            self.config["default_model"] = model_settings.name
        
        self._save_config()
        self._save_api_keys()
    
    def get_model(self, name: str) -> Optional[ModelSettings]:
        for model in self.config["models"]:
            if model["name"] == name:
                return ModelSettings(**model)
        return None
    
    def chat_completion(
        self,
        model_name: str,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: int = 2048
    ) -> Tuple[str, int, int, float]:
        model_settings = self.get_model(model_name)
        if not model_settings:
            raise ValueError("Model not found")
        
        api_key = self.api_keys.get(model_name)
        if not api_key:
            raise ValueError("API key not found for this model")
        
        model_type = next(
            (k for k, v in MODELS.items() if v["base_url"] in model_settings.api_url),
            None
        )
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model_settings.model_id,
            "messages": messages,
            "temperature": temperature if temperature is not None else model_settings.temperature,
            "top_p": top_p if top_p is not None else model_settings.top_p,
            "max_tokens": max_tokens
        }
        
        start_time = time.time()
        with Progress() as progress:
            task = progress.add_task("[cyan]Generating response...", total=None)
            
            response = requests.post(
                f"{MODELS[model_type]['base_url']}/chat/completions",
                headers=headers,
                json=payload,
                stream=True
            )
            
            if response.status_code != 200:
                progress.stop()
                raise ValueError(f"API request failed: {response.text}")
            
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            input_tokens = data["usage"]["prompt_tokens"]
            output_tokens = data["usage"]["completion_tokens"]
            
            progress.update(task, completed=100)
            progress.stop()
        
        duration = time.time() - start_time
        tokens_per_second = output_tokens / duration if duration > 0 else 0
        cost = self.estimate_cost(input_tokens, output_tokens, model_settings)
        
        return content, input_tokens, output_tokens, cost, tokens_per_second

class Conversation:
    def __init__(self, name: str, model_name: str):
        self.name = name
        self.model_name = model_name
        self.messages = []
        self.created_at = time.time()
        self.updated_at = time.time()
        self.token_count = 0
        self.cost = 0.0
    
    def add_message(self, role: str, content: str, tokens: int, cost: float):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": time.time(),
            "tokens": tokens
        })
        self.token_count += tokens
        self.cost += cost
        self.updated_at = time.time()
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "model_name": self.model_name,
            "messages": self.messages,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "token_count": self.token_count,
            "cost": self.cost
        }

class ConversationManager:
    def __init__(self, client: LLMClient):
        self.client = client
        self.conversations = {}
        self.current_conversation = None
    
    def create_conversation(self, name: str, model_name: str) -> Conversation:
        if name in self.conversations:
            raise ValueError("Conversation with this name already exists")
        
        if not self.client.get_model(model_name):
            raise ValueError("Model not found")
        
        conversation = Conversation(name, model_name)
        self.conversations[name] = conversation
        self.current_conversation = conversation
        return conversation
    
    def delete_conversation(self, name: str):
        if name not in self.conversations:
            raise ValueError("Conversation not found")
        
        if self.current_conversation and self.current_conversation.name == name:
            self.current_conversation = None
        
        del self.conversations[name]
    
    def rename_conversation(self, old_name: str, new_name: str):
        if old_name not in self.conversations:
            raise ValueError("Conversation not found")
        
        if new_name in self.conversations:
            raise ValueError("Conversation with new name already exists")
        
        conversation = self.conversations[old_name]
        conversation.name = new_name
        del self.conversations[old_name]
        self.conversations[new_name] = conversation
        
        if self.current_conversation and self.current_conversation.name == old_name:
            self.current_conversation = conversation
    
    def archive_conversation(self, name: str, archive_path: str = "conversations_archive"):
        if name not in self.conversations:
            raise ValueError("Conversation not found")
        
        os.makedirs(archive_path, exist_ok=True)
        conversation = self.conversations[name]
        
        with open(f"{archive_path}/{name}.json", "w") as f:
            json.dump(conversation.to_dict(), f, indent=2)
        
        self.delete_conversation(name)
    
    def load_conversation(self, name: str, archive_path: str = "conversations_archive"):
        file_path = f"{archive_path}/{name}.json"
        if not os.path.exists(file_path):
            raise ValueError("Archived conversation not found")
        
        with open(file_path, "r") as f:
            data = json.load(f)
        
        conversation = Conversation(data["name"], data["model_name"])
        conversation.messages = data["messages"]
        conversation.created_at = data["created_at"]
        conversation.updated_at = data["updated_at"]
        conversation.token_count = data["token_count"]
        conversation.cost = data["cost"]
        
        self.conversations[name] = conversation
        self.current_conversation = conversation
        return conversation

class LLMApplication:
    def __init__(self):
        self.client = LLMClient()
        self.conversation_manager = ConversationManager(self.client)
        self.console = Console()
        self.running = True
    
    def display_main_menu(self):
        self.console.clear()
        self.console.print(Panel.fit("LLM API Client", style="bold blue"))
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Option", style="dim")
        table.add_column("Description")
        
        table.add_row("1", "Start new conversation")
        table.add_row("2", "Continue conversation")
        table.add_row("3", "Manage models")
        table.add_row("4", "View conversation history")
        table.add_row("5", "Settings")
        table.add_row("q", "Quit")
        
        self.console.print(table)
    
    def prompt_input(self, text: str) -> str:
        return self.console.input(f"[bold green]»[/] {text}: ")
    
    def display_conversation(self, conversation: Conversation):
        self.console.clear()
        self.console.print(Panel.fit(f"Conversation: {conversation.name} (Model: {conversation.model_name})", 
                                   style="bold blue"))
        
        for msg in conversation.messages:
            style = "bold green" if msg["role"] == "user" else "bold cyan"
            self.console.print(f"[{style}]{msg['role'].upper()}:[/] {msg['content']}")
            self.console.print(f"[dim]Tokens: {msg['tokens']} | {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(msg['timestamp']))}[/]")
            self.console.print()
        
        self.console.print(Panel.fit(f"Total tokens: {conversation.token_count} | Estimated cost: ${conversation.cost:.4f}",
                                   style="bold yellow"))
    
    def run(self):
        while self.running:
            self.display_main_menu()
            choice = self.prompt_input("Select an option").lower()
            
            if choice == "1":
                self.handle_new_conversation()
            elif choice == "2":
                self.handle_continue_conversation()
            elif choice == "3":
                self.handle_manage_models()
            elif choice == "4":
                self.handle_view_history()
            elif choice == "5":
                self.handle_settings()
            elif choice == "q":
                self.running = False
            else:
                self.console.print("[red]Invalid option[/]")
                time.sleep(1)
    
    def handle_new_conversation(self):
        name = self.prompt_input("Enter conversation name")
        model_name = self.prompt_input("Enter model name")
        
        try:
            conversation = self.conversation_manager.create_conversation(name, model_name)
            self.handle_chat_loop(conversation)
        except ValueError as e:
            self.console.print(f"[red]Error: {e}[/]")
            time.sleep(2)
    
    def handle_chat_loop(self, conversation: Conversation):
        while True:
            self.display_conversation(conversation)
            prompt = self.prompt_input("Your message (or '/exit' to quit)")
            
            if prompt.lower() == "/exit":
                break
            
            try:
                messages = [{"role": "user", "content": prompt}]
                response, input_tokens, output_tokens, cost, speed = self.client.chat_completion(
                    conversation.model_name,
                    messages
                )
                
                conversation.add_message("user", prompt, input_tokens, 0)
                conversation.add_message("assistant", response, output_tokens, cost)
                
                self.console.print(f"[dim]Speed: {speed:.1f} tokens/s | Cost: ${cost:.4f}[/]")
                time.sleep(0.5)
            except Exception as e:
                self.console.print(f"[red]Error: {e}[/]")
                time.sleep(2)

    def handle_continue_conversation(self):
        if not self.conversation_manager.conversations:
            self.console.print("[yellow]No conversations available[/]")
            time.sleep(1)
            return

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim")
        table.add_column("Name")
        table.add_column("Model")
        table.add_column("Messages")
        table.add_column("Last Updated")
        
        for i, (name, conv) in enumerate(self.conversation_manager.conversations.items(), 1):
            table.add_row(
                str(i),
                name,
                conv.model_name,
                str(len(conv.messages)),
                time.strftime('%Y-%m-%d %H:%M', time.localtime(conv.updated_at))
            )
        
        self.console.print(table)
        choice = self.prompt_input("Select conversation number or 'b' to go back")
        
        if choice.lower() == 'b':
            return
            
        try:
            idx = int(choice) - 1
            conv_name = list(self.conversation_manager.conversations.keys())[idx]
            conversation = self.conversation_manager.conversations[conv_name]
            self.handle_chat_loop(conversation)
        except (ValueError, IndexError):
            self.console.print("[red]Invalid selection[/]")
            time.sleep(1)

    def handle_manage_models(self):
        while True:
            self.console.clear()
            self.console.print(Panel.fit("Model Management", style="bold blue"))
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Option", style="dim")
            table.add_column("Description")
            
            table.add_row("1", "Add new model")
            table.add_row("2", "Remove model")
            table.add_row("3", "List models")
            table.add_row("b", "Back to main menu")
            
            self.console.print(table)
            choice = self.prompt_input("Select an option").lower()
            
            if choice == "1":
                self.handle_add_model()
            elif choice == "2":
                self.handle_remove_model()
            elif choice == "3":
                self.handle_list_models()
            elif choice == "b":
                break
            else:
                self.console.print("[red]Invalid option[/]")
                time.sleep(1)
    
    def handle_add_model(self):
        self.console.print(Panel.fit("Available Model Types:", style="bold blue"))
        for i, (name, details) in enumerate(MODELS.items(), 1):
            self.console.print(f"{i}. {name} ({details['base_url']})")
        
        try:
            model_type_idx = int(self.prompt_input("Select model type (number)")) - 1
            model_type = list(MODELS.keys())[model_type_idx]
            
            name = self.prompt_input("Enter display name for this model")
            model_id = self.prompt_input("Enter model ID (e.g. 'gpt-4')")
            api_key = self.prompt_input("Enter API key")
            input_cost = float(self.prompt_input("Enter input cost per 1M tokens"))
            output_cost = float(self.prompt_input("Enter output cost per 1M tokens"))
            
            model_settings = ModelSettings(
                name=name,
                api_url=MODELS[model_type]["base_url"],
                model_id=model_id,
                input_cost=input_cost,
                output_cost=output_cost
            )
            
            self.client.add_model(model_settings, api_key)
            self.console.print("[green]Model added successfully![/]")
            time.sleep(1)
        except (ValueError, IndexError) as e:
            self.console.print(f"[red]Error: {e}[/]")
            time.sleep(2)
    
    def handle_remove_model(self):
        models = self.client.config.get("models", [])
        if not models:
            self.console.print("[yellow]No models configured[/]")
            time.sleep(1)
            return
            
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim")
        table.add_column("Name")
        table.add_column("API URL")
        table.add_column("Model ID")
        
        for i, model in enumerate(models, 1):
            table.add_row(
                str(i),
                model["name"],
                model["api_url"],
                model["model_id"]
            )
        
        self.console.print(table)
        choice = self.prompt_input("Select model to remove or 'b' to go back")
        
        if choice.lower() == 'b':
            return
            
        try:
            idx = int(choice) - 1
            model_name = models[idx]["name"]
            
            # Confirm deletion
            confirm = self.prompt_input(f"Confirm delete model '{model_name}'? (y/n)").lower()
            if confirm == 'y':
                self.client.config["models"].pop(idx)
                if self.client.config["default_model"] == model_name:
                    self.client.config["default_model"] = None
                self.client._save_config()
                self.console.print("[green]Model removed successfully![/]")
            else:
                self.console.print("[yellow]Cancelled[/]")
            time.sleep(1)
        except (ValueError, IndexError):
            self.console.print("[red]Invalid selection[/]")
            time.sleep(1)
    
    def handle_list_models(self):
        models = self.client.config.get("models", [])
        if not models:
            self.console.print("[yellow]No models configured[/]")
            time.sleep(1)
            return
            
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Name")
        table.add_column("API URL")
        table.add_column("Model ID")
        table.add_column("Input Cost")
        table.add_column("Output Cost")
        
        for model in models:
            table.add_row(
                model["name"],
                model["api_url"],
                model["model_id"],
                f"${model['input_cost']}/1M",
                f"${model['output_cost']}/1M"
            )
        
        self.console.print(table)
        self.prompt_input("Press Enter to continue")
    
    def handle_view_history(self):
        archive_path = "conversations_archive"
        if not os.path.exists(archive_path) or not os.listdir(archive_path):
            self.console.print("[yellow]No archived conversations found[/]")
            time.sleep(1)
            return

        # Get all archived conversation files
        conv_files = [f for f in os.listdir(archive_path) if f.endswith('.json')]
        if not conv_files:
            self.console.print("[yellow]No archived conversations found[/]")
            time.sleep(1)
            return

        while True:
            self.console.clear()
            self.console.print(Panel.fit("Archived Conversations", style="bold blue"))
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("#", style="dim")
            table.add_column("Name")
            table.add_column("Model")
            table.add_column("Messages")
            table.add_column("Tokens")
            table.add_column("Cost")
            table.add_column("Last Updated")
            
            for i, conv_file in enumerate(conv_files, 1):
                with open(f"{archive_path}/{conv_file}", "r") as f:
                    data = json.load(f)
                table.add_row(
                    str(i),
                    data["name"],
                    data["model_name"],
                    str(len(data["messages"])),
                    str(data["token_count"]),
                    f"${data['cost']:.4f}",
                    time.strftime('%Y-%m-%d %H:%M', time.localtime(data["updated_at"]))
                )
            
            self.console.print(table)
            
            self.console.print("\nOptions:")
            self.console.print("1. View conversation details")
            self.console.print("2. Load conversation")
            self.console.print("3. Delete archived conversation")
            self.console.print("b. Back to main menu")
            
            choice = self.prompt_input("Select an option").lower()
            
            if choice == "1":
                self.handle_view_conversation_details(archive_path, conv_files)
            elif choice == "2":
                self.handle_load_archived_conversation(archive_path, conv_files)
            elif choice == "3":
                self.handle_delete_archived_conversation(archive_path, conv_files)
            elif choice == "b":
                break
            else:
                self.console.print("[red]Invalid option[/]")
                time.sleep(1)
    
    def handle_view_conversation_details(self, archive_path: str, conv_files: List[str]):
        choice = self.prompt_input("Select conversation number to view details")
        try:
            idx = int(choice) - 1
            conv_file = conv_files[idx]
            
            with open(f"{archive_path}/{conv_file}", "r") as f:
                data = json.load(f)
            
            self.console.clear()
            self.console.print(Panel.fit(f"Conversation Details: {data['name']}", style="bold blue"))
            
            # Display basic info
            info_table = Table(show_header=False)
            info_table.add_column("Field", style="dim")
            info_table.add_column("Value")
            
            info_table.add_row("Model", data["model_name"])
            info_table.add_row("Total Messages", str(len(data["messages"])))
            info_table.add_row("Total Tokens", str(data["token_count"]))
            info_table.add_row("Estimated Cost", f"${data['cost']:.4f}")
            info_table.add_row("Created", time.strftime('%Y-%m-%d %H:%M', time.localtime(data["created_at"])))
            info_table.add_row("Last Updated", time.strftime('%Y-%m-%d %H:%M', time.localtime(data["updated_at"])))
            
            self.console.print(info_table)
            self.prompt_input("Press Enter to continue")
            
            # Display message samples
            self.console.print(Panel.fit("Message Samples", style="bold blue"))
            for msg in data["messages"][:5]:  # Show first 5 messages
                style = "bold green" if msg["role"] == "user" else "bold cyan"
                self.console.print(f"[{style}]{msg['role'].upper()}:[/] {msg['content']}")
                self.console.print(f"[dim]Tokens: {msg['tokens']} | {time.strftime('%Y-%m-%d %H:%M', time.localtime(msg['timestamp']))}[/]")
                self.console.print()
            
            self.prompt_input("Press Enter to continue")
        except (ValueError, IndexError):
            self.console.print("[red]Invalid selection[/]")
            time.sleep(1)
    
    def handle_load_archived_conversation(self, archive_path: str, conv_files: List[str]):
        choice = self.prompt_input("Select conversation number to load")
        try:
            idx = int(choice) - 1
            conv_name = os.path.splitext(conv_files[idx])[0]
            
            conversation = self.conversation_manager.load_conversation(conv_name, archive_path)
            self.console.print(f"[green]Loaded conversation: {conv_name}[/]")
            time.sleep(1)
            self.handle_chat_loop(conversation)
        except (ValueError, IndexError) as e:
            self.console.print(f"[red]Error: {e}[/]")
            time.sleep(1)
    
    def handle_delete_archived_conversation(self, archive_path: str, conv_files: List[str]):
        choice = self.prompt_input("Select conversation number to delete")
        try:
            idx = int(choice) - 1
            conv_file = conv_files[idx]
            conv_name = os.path.splitext(conv_file)[0]
            
            confirm = self.prompt_input(f"Confirm delete '{conv_name}'? (y/n)").lower()
            if confirm == 'y':
                os.remove(f"{archive_path}/{conv_file}")
                self.console.print(f"[green]Deleted conversation: {conv_name}[/]")
            else:
                self.console.print("[yellow]Cancelled[/]")
            time.sleep(1)
        except (ValueError, IndexError, OSError) as e:
            self.console.print(f"[red]Error: {e}[/]")
            time.sleep(1)
    
    def handle_settings(self):
        while True:
            self.console.clear()
            self.console.print(Panel.fit("Settings", style="bold blue"))
            
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Option", style="dim")
            table.add_column("Description")
            
            table.add_row("1", "Change default model")
            table.add_row("2", "View API keys")
            table.add_row("3", "Reset all data")
            table.add_row("b", "Back to main menu")
            
            self.console.print(table)
            choice = self.prompt_input("Select an option").lower()
            
            if choice == "1":
                self.handle_change_default_model()
            elif choice == "2":
                self.handle_view_api_keys()
            elif choice == "3":
                self.handle_reset_data()
            elif choice == "b":
                break
            else:
                self.console.print("[red]Invalid option[/]")
                time.sleep(1)
    
    def handle_change_default_model(self):
        models = self.client.config.get("models", [])
        if not models:
            self.console.print("[yellow]No models configured[/]")
            time.sleep(1)
            return
            
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="dim")
        table.add_column("Name")
        table.add_column("Current Default", style="bold green" if not self.client.config["default_model"] else None)
        
        for i, model in enumerate(models, 1):
            is_default = model["name"] == self.client.config["default_model"]
            table.add_row(
                str(i),
                model["name"],
                "✓" if is_default else ""
            )
        
        self.console.print(table)
        choice = self.prompt_input("Select new default model or 'b' to go back")
        
        if choice.lower() == 'b':
            return
            
        try:
            idx = int(choice) - 1
            model_name = models[idx]["name"]
            self.client.config["default_model"] = model_name
            self.client._save_config()
            self.console.print(f"[green]Default model set to: {model_name}[/]")
            time.sleep(1)
        except (ValueError, IndexError):
            self.console.print("[red]Invalid selection[/]")
            time.sleep(1)
    
    def handle_view_api_keys(self):
        if not self.client.api_keys:
            self.console.print("[yellow]No API keys stored[/]")
            time.sleep(1)
            return
            
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Model Name")
        table.add_column("API Key (truncated)")
        
        for name, key in self.client.api_keys.items():
            truncated = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "****"
            table.add_row(name, truncated)
        
        self.console.print(table)
        self.prompt_input("Press Enter to continue")
    
    def handle_reset_data(self):
        confirm = self.prompt_input("WARNING: This will delete ALL data. Confirm? (y/n)").lower()
        if confirm != 'y':
            self.console.print("[yellow]Cancelled[/]")
            time.sleep(1)
            return
            
        try:
            # Delete config files
            for f in [CONFIG_FILE, API_KEYS_FILE, "encryption.key"]:
                if os.path.exists(f):
                    os.remove(f)
            
            # Delete conversation archives
            archive_path = "conversations_archive"
            if os.path.exists(archive_path):
                for f in os.listdir(archive_path):
                    os.remove(os.path.join(archive_path, f))
                os.rmdir(archive_path)
            
            # Reset in-memory state
            self.client = LLMClient()
            self.conversation_manager = ConversationManager(self.client)
            
            self.console.print("[green]All data has been reset[/]")
            time.sleep(1)
        except Exception as e:
            self.console.print(f"[red]Error: {e}[/]")
            time.sleep(1)

# TODO: Implement remaining features:
# - Advanced prompt engineering
# - Security features
# - Cost tracking
# - Multi-model comparisons

if __name__ == "__main__":
    app = LLMApplication()
    app.run()
