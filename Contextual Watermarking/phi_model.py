import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor, LogitsProcessorList
from scipy.stats import norm
from typing import List
import hashlib
from dataclasses import dataclass
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class WatermarkConfig:
    gamma: float = 0.7
    delta: float = 10.0
    context_window: int = 3
    min_tokens: int = 50
    secret_key: str = "my_secure_secret_123"
    z_threshold: float = 4.5
    dynamic_threshold: bool = True

class SecureContextHasher:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key.encode('utf-8')

    def __call__(self, context: torch.Tensor) -> int:
        context_bytes = b''.join([int(t.item()).to_bytes(4, 'big', signed=True) for t in context])
        hmac = hashlib.blake2b(key=self.secret_key, digest_size=16)
        hmac.update(context_bytes)
        return int.from_bytes(hmac.digest(), 'big')

class RobustWatermarkProcessor(LogitsProcessor):
    def __init__(self, config: WatermarkConfig, vocab: List[int]):
        self.config = config
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.hasher = SecureContextHasher(config.secret_key)
        self.green_cache = {}
        self.red_cache = {}

    def _get_token_lists(self, context: torch.Tensor):
        context_hash = hash(context.cpu().numpy().tobytes())
        if context_hash in self.green_cache:
            return self.green_cache[context_hash], self.red_cache[context_hash]
        seed = self.hasher(context)
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(self.vocab_size)
        green_size = int(self.config.gamma * self.vocab_size)
        red_size = int(green_size * 0.3)
        green_list = permutation[:green_size]
        red_list = permutation[green_size:green_size+red_size]
        if len(self.green_cache) > 10000:
            oldest_key = next(iter(self.green_cache))
            del self.green_cache[oldest_key]
            del self.red_cache[oldest_key]
        self.green_cache[context_hash] = green_list
        self.red_cache[context_hash] = red_list
        return green_list, red_list

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        batch_size, seq_len = input_ids.shape
        h = self.config.context_window
        for batch_idx in range(batch_size):
            if seq_len < h:
                continue
            context = input_ids[batch_idx, -h:].cpu()
            green_list, red_list = self._get_token_lists(context)
            scores[batch_idx][green_list] += self.config.delta
            scores[batch_idx][red_list] -= self.config.delta * 0.7
        return scores

class WatermarkEncoder:
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct", config: WatermarkConfig = WatermarkConfig()):
        self.device = device
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.processor = RobustWatermarkProcessor(config, list(self.tokenizer.get_vocab().values()))

    def generate(self, prompt: str, max_length: int = 300, temperature: float = 0.6) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + max_length,
            logits_processor=LogitsProcessorList([self.processor]),
            do_sample=True,
            top_p=0.95,
            temperature=temperature,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class WatermarkDetector:
    def __init__(self, tokenizer: AutoTokenizer, config: WatermarkConfig = WatermarkConfig()):
        self.tokenizer = tokenizer
        self.config = config
        self.vocab = list(tokenizer.get_vocab().values())
        self.vocab_size = len(self.vocab)
        self.hasher = SecureContextHasher(config.secret_key)
        self.results_cache = {}

    def _calculate_dynamic_threshold(self, token_length: int) -> float:
        base = self.config.z_threshold
        if not self.config.dynamic_threshold:
            return base
        if token_length < 100:
            return base * 2.0  # Higher threshold for short texts
        elif token_length > 500:
            return base * 0.8
        return base

    def detect(self, text: str) -> dict:
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        if text_hash in self.results_cache:
            return self.results_cache[text_hash]
        tokens = self.tokenizer.encode(text, return_tensors="pt")[0]
        T = len(tokens)
        if T < self.config.min_tokens:
            return {
                "is_watermarked": False,
                "confidence": 0.0,
                "message": f"Text too short ({T} tokens < {self.config.min_tokens} min)"
            }
        green_counts = []
        h = self.config.context_window
        for i in range(h, T):
            context = tokens[i-h:i]
            seed = self.hasher(context)
            rng = np.random.default_rng(seed)
            permutation = rng.permutation(self.vocab_size)
            green_list = permutation[:int(self.config.gamma*self.vocab_size)]
            green_counts.append(tokens[i] in green_list)
        observed = sum(green_counts)
        expected = self.config.gamma * (T - h)
        std = np.sqrt((T - h) * self.config.gamma * (1 - self.config.gamma))
        z_score = (observed - expected) / std if std > 0 else 0
        threshold = self._calculate_dynamic_threshold(T)
        p_value = 1 - norm.cdf(z_score)
        chunk_size = max(1, (T - h) // 10)
        chunk_means = [
            np.mean(green_counts[i*chunk_size:(i+1)*chunk_size])
            for i in range(10)
            if i*chunk_size < len(green_counts)
        ]
        result = {
            "z_score": z_score,
            "p_value": p_value,
            "confidence": min(100, max(0, (1 - p_value)*100)),
            "is_watermarked": z_score > threshold,
            "threshold_used": threshold,
            "temporal_variation": np.std(chunk_means) if chunk_means else 0,
            "green_rate": observed/(T - h),
            "expected_rate": self.config.gamma,
            "tokens_analyzed": T - h
        }
        self.results_cache[text_hash] = result
        return result

if __name__ == "__main__":
    config = WatermarkConfig(
        gamma=0.7,
        delta=10.0,
        context_window=3,
        min_tokens=50,
        secret_key="my_secure_secret_123",
        z_threshold=4.5
    )

    encoder = WatermarkEncoder(config=config)
    detector = WatermarkDetector(encoder.tokenizer, config=config)

    prompt = "Explain quantum computing in simple terms."
    watermarked_text = encoder.generate(prompt, max_length=300, temperature=0.6)
    print("Generated Text:\n", watermarked_text[:500] + "...\n")

    detection = detector.detect(watermarked_text)
    print("Detection Results:")
    print(f"Z-score: {detection['z_score']:.2f} (Threshold: {detection['threshold_used']:.2f})")
    print(f"Confidence: {detection['confidence']:.1f}%")
    print(f"Watermarked: {detection['is_watermarked']}")

    normal_text = (" Explain quantum computing in simple terms. Ensuring the explanation is accessible for someone with basic knowledge of classical computing. Quantitative Analysis: Using the data provided in the document, calculate the potential reduction in computational time if a quantum computer with 5 qubit operations per cycle was able to perform a complex task that a conventional computer with 4 operations per cycle requires 1,00,700,200 operations, and each operation takes approximately 10 nanoseconds. Quan...")
    normal_detection = detector.detect(normal_text)
    print("\nNormal Text Detection:")
    print(f"Z-score: {normal_detection['z_score']:.2f}")
    print(f"Watermarked: {normal_detection['is_watermarked']}")
