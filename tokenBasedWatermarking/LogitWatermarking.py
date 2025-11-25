import argparse
import logging
import os
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from scipy.stats import norm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")

class HashNet(nn.Module):
    def __init__(self, input_size: int = 4, hidden_size: int = 256):
        super(HashNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.net(x) * (2**31)).int()

class WatermarkLogitsProcessor(LogitsProcessor):
    def __init__(self, vocab: list, gamma: float, delta: float, h: int, hash_net: HashNet, base_key: int = 982451653):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.gamma = gamma
        self.delta = delta
        self.h = h
        self.hash_net = hash_net
        self.base_key = base_key
        self.generator = torch.Generator(device=device)

    def get_green_list(self, prev_tokens: torch.Tensor) -> torch.Tensor:
        norm_tokens = prev_tokens.float() / self.vocab_size
        seed = int(self.hash_net(norm_tokens.unsqueeze(0)).item() + self.base_key) & 0xFFFFFFFF
        self.generator.manual_seed(seed)
        green_size = int(self.gamma * self.vocab_size)
        return torch.randperm(self.vocab_size, device=device, generator=self.generator)[:green_size]

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        for b_idx in range(input_ids.shape[0]):
            start_idx = max(0, input_ids.size(1) - self.h)
            prev_tokens = input_ids[b_idx, start_idx:]
            if prev_tokens.size(0) < self.h:
                prev_tokens = torch.cat([torch.zeros(self.h - prev_tokens.size(0), dtype=torch.long, device=device), prev_tokens])
            green_list = self.get_green_list(prev_tokens)
            scores[b_idx][green_list] += self.delta
        return scores

class WatermarkEncoder:
    def __init__(self, model_name: str = "facebook/opt-1.3b", gamma: float = 0.25, delta: float = 2.0, h: int = 4, base_key: int = 982451653):
        self.device = device
        self.gamma = gamma
        self.delta = delta
        self.h = h
        self.base_key = base_key
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.vocab = list(self.tokenizer.get_vocab().values())
        
        self.hash_net = HashNet(input_size=h).to(device)
        hashnet_path = "/content/secret_hash_net.pt"
        if not os.path.exists(hashnet_path):
            torch.save(self.hash_net.state_dict(), hashnet_path)
            logging.info(f"Initialized and saved HashNet to {hashnet_path}")
        else:
            self.hash_net.load_state_dict(torch.load(hashnet_path, map_location=device))
            self.hash_net.eval()
            logging.info(f"Loaded HashNet from {hashnet_path}")

    def generate(self, prompt: str, max_length: int = 50) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
        watermark_processor = WatermarkLogitsProcessor(
            vocab=self.vocab, gamma=self.gamma, delta=self.delta, h=self.h,
            hash_net=self.hash_net, base_key=self.base_key
        )
        outputs = self.model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + max_length,
            logits_processor=LogitsProcessorList([watermark_processor]),
            do_sample=True,
            top_p=0.95,
            temperature=0.7
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class WatermarkDecoder:
    def __init__(self, tokenizer, vocab: list, gamma: float = 0.25, h: int = 4, hash_net: HashNet = None, base_key: int = 982451653, z_threshold: float = 2.5):
        self.tokenizer = tokenizer
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.gamma = gamma
        self.h = h
        self.hash_net = hash_net
        self.base_key = base_key
        self.z_threshold = z_threshold
        self.device = device
        self.generator = torch.Generator(device=device)

    def get_green_list(self, prev_tokens: torch.Tensor) -> torch.Tensor:
        norm_tokens = prev_tokens.float() / self.vocab_size
        seed = int(self.hash_net(norm_tokens.unsqueeze(0)).item() + self.base_key) & 0xFFFFFFFF
        self.generator.manual_seed(seed)
        green_size = int(self.gamma * self.vocab_size)
        return torch.randperm(self.vocab_size, device=self.device, generator=self.generator)[:green_size]

    def detect(self, text: str) -> dict:
        tokens = self.tokenizer.encode(text, return_tensors="pt").to(self.device)[0]
        T = len(tokens)
        if T <= self.h:
            return {"is_watermarked": False, "z_score": 0.0, "message": "Text too short for detection"}
        
        green_count = 0
        for t in range(self.h, T):
            prev_tokens = tokens[t - self.h:t]
            green_list = self.get_green_list(prev_tokens)
            if tokens[t].item() in green_list:
                green_count += 1
        
        T_valid = T - self.h
        expected = self.gamma * T_valid
        variance = T_valid * self.gamma * (1 - self.gamma)
        z_score = (green_count - expected) / np.sqrt(variance) if variance > 0 else 0
        return {
            "is_watermarked": z_score > self.z_threshold,
            "z_score": z_score,
            "message": f"Detected {green_count}/{T_valid} green tokens"
        }

def main(argv=None):
    parser = argparse.ArgumentParser(description="Watermark Embedding and Detection", add_help=False)
    parser.add_argument("--prompt", type=str, default="", help="Prompt to generate watermarked text")
    parser.add_argument("--text", type=str, default="", help="Text to check for watermark")
    parser.add_argument("--max_length", type=int, default=50, help="Max length of generated text")
    
    args, unknown = parser.parse_known_args(argv if argv is not None else [])

    encoder = WatermarkEncoder()
    decoder_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-1.3b")
    vocab = list(decoder_tokenizer.get_vocab().values())
    hash_net = HashNet(input_size=4).to(device)
    hash_net.load_state_dict(torch.load("/content/secret_hash_net.pt", map_location=device))
    hash_net.eval()
    decoder = WatermarkDecoder(decoder_tokenizer, vocab, hash_net=hash_net)

    if not args.prompt and not args.text:
        print("Enter a mode: 'encode' to generate watermarked text, 'decode' to detect a watermark")
        mode = input("Mode: ").strip().lower()
        if mode == "encode":
            args.prompt = input("Enter your prompt: ").strip()
        elif mode == "decode":
            args.text = input("Enter text to detect watermark: ").strip()
        else:
            print("Invalid mode. Use 'encode' or 'decode'.")
            return

    if args.prompt:
        logging.info("Generating watermarked text...")
        watermarked_text = encoder.generate(args.prompt, max_length=args.max_length)
        print("\n=== Watermarked Text ===")
        print(watermarked_text)
        
        detection = decoder.detect(watermarked_text)
        print("\n=== Detection Results ===")
        print(f"Watermark Detected: {detection['is_watermarked']}")
        print(f"Z-Score: {detection['z_score']:.2f}")
        print(f"Details: {detection['message']}")

    if args.text:
        logging.info("Detecting watermark in provided text...")
        detection = decoder.detect(args.text)
        print("\n=== Detection Results ===")
        print(f"Watermark Detected: {detection['is_watermarked']}")
        print(f"Z-Score: {detection['z_score']:.2f}")
        print(f"Details: {detection['message']}")

if __name__ == "__main__":
    import sys
    main(sys.argv[1:])