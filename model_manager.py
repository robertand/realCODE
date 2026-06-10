import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
)
import gc
import logging
import os
import sys


# Setează LD_LIBRARY_PATH pentru CUDA 13 (nvidia pypi packages)
_cuda_fixed = False
for p in sys.path:
    nv = os.path.join(p, "nvidia")
    if os.path.isdir(nv):
        for d in os.listdir(nv):
            lib_dir = os.path.join(nv, d, "lib")
            if os.path.isdir(lib_dir):
                existing = os.environ.get("LD_LIBRARY_PATH", "")
                if lib_dir not in existing:
                    os.environ["LD_LIBRARY_PATH"] = f"{lib_dir}:{existing}" if existing else lib_dir
                    _cuda_fixed = True
if _cuda_fixed:
    print(f"[startup] Fixed CUDA library path")


logger = logging.getLogger(__name__)

_BITSANDBYTES_AVAILABLE = False
try:
    import bitsandbytes
    _BITSANDBYTES_AVAILABLE = True
except (ImportError, FileNotFoundError):
    pass


class ModelManager:
    def __init__(self, model_id: str, quantization: str = "4bit", device: str = "cuda"):
        self.model_id = model_id
        self.quantization = quantization
        self.device = device if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def _fix_config_if_needed(self):
        """Detect and fix missing model_type in config for custom Qwen models"""
        try:
            config = AutoConfig.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            )
            
            if not hasattr(config, 'model_type') or not config.model_type:
                model_id_lower = self.model_id.lower()
                if 'qwen' in model_id_lower:
                    config.model_type = 'qwen2'
                    print(f"[fix] Inferred model_type='qwen2' for {self.model_id}")
                elif 'llama' in model_id_lower:
                    config.model_type = 'llama'
                elif 'gemma' in model_id_lower:
                    config.model_type = 'gemma'
                else:
                    config.model_type = 'qwen2'
                    print(f"[fix] Using default model_type='qwen2' for {self.model_id}")
                
            return config
        except Exception as e:
            print(f"[warning] Could not auto-fix config: {e}")
            return None

    def _apply_compat_patches(self):
        """Apply compatibility patches for newer transformers versions"""
        import transformers.utils.import_utils as iu
        if not hasattr(iu, 'is_torch_fx_available'):
            def is_torch_fx_available():
                try:
                    import torch.fx
                    return True
                except ImportError:
                    return False
            iu.is_torch_fx_available = is_torch_fx_available
            print("[compat] Patched is_torch_fx_available")

        # transformers 4.41+ moved/removed some attributes from DynamicCache
        try:
            import transformers.cache_utils as cu
            target_classes = []
            for name in ["Cache", "DynamicCache"]:
                if hasattr(cu, name):
                    cls = getattr(cu, name)
                    if cls not in target_classes:
                        target_classes.append(cls)

            # Also check main transformers namespace
            import transformers
            for name in ["Cache", "DynamicCache"]:
                if hasattr(transformers, name):
                    cls = getattr(transformers, name)
                    if cls not in target_classes:
                        target_classes.append(cls)

            for cls in target_classes:
                # Patch seen_tokens
                if not hasattr(cls, 'seen_tokens'):
                    def get_seen_tokens(self):
                        return self.get_seq_length()
                    setattr(cls, 'seen_tokens', property(get_seen_tokens))
                    print(f"[compat] Patched {cls.__name__}.seen_tokens")

                # Patch get_max_length
                if not hasattr(cls, 'get_max_length'):
                    def get_max_length(self):
                        # DeepSeek models expect None if no limit is set, or the actual limit.
                        # Returning 2048 was causing size mismatch errors.
                        return getattr(self, 'max_cache_length', None)
                    setattr(cls, 'get_max_length', get_max_length)
                    print(f"[compat] Patched {cls.__name__}.get_max_length")

                # Patch get_usable_length
                if not hasattr(cls, 'get_usable_length'):
                    def get_usable_length(self, sequence_length, layer_idx=None):
                        max_length = self.get_max_length()
                        past_length = self.get_seq_length()
                        if max_length is None:
                            return past_length
                        return min(past_length, max_length - sequence_length)
                    setattr(cls, 'get_usable_length', get_usable_length)
                    print(f"[compat] Patched {cls.__name__}.get_usable_length")
        except Exception as e:
            print(f"[compat] Failed to apply DynamicCache patches: {e}")

    def load(self):
        if self._loaded:
            return

        self._apply_compat_patches()

        print(f"Loading model: {self.model_id}")
        print(f"Quantization: {self.quantization}")
        print(f"Device: {self.device}")
        
        # Verifică memoria GPU disponibilă
        if torch.cuda.is_available():
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            free_gb = free_memory / 1e9
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU memory: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
            
            if free_gb < 20 and "14B" in self.model_id and self.quantization == "none":
                print(f"[warning] Only {free_gb:.1f}GB free, 14B model needs ~28GB")
                print("[warning] Switching to 4bit quantization automatically")
                self.quantization = "4bit"

        kw = {}

        # Try to fix missing model_type in config
        fixed_config = self._fix_config_if_needed()
        if fixed_config:
            kw["config"] = fixed_config

        if self.device == "cuda":
            if self.quantization in ("4bit", "8bit"):
                if not _BITSANDBYTES_AVAILABLE:
                    print("bitsandbytes not installed. Falling back to no quantization.")
                    print("Install with: pip install bitsandbytes")
                    self.quantization = "none"
                    kw["torch_dtype"] = torch.bfloat16
                    kw["device_map"] = "auto"
                else:
                    if self.quantization == "4bit":
                        kw["quantization_config"] = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.bfloat16,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_quant_type="nf4",
                        )
                    elif self.quantization == "8bit":
                        kw["quantization_config"] = BitsAndBytesConfig(
                            load_in_8bit=True
                        )
                    kw["device_map"] = "auto"
                    kw["torch_dtype"] = torch.bfloat16
                    
                    # FIX: Folosește string cu unitate, nu integer
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    max_memory_bytes = int(total_memory * 0.95)
                    # Transformă în string cu unitate GB
                    max_memory_gb = max_memory_bytes / (1024**3)
                    kw["max_memory"] = {0: f"{max_memory_gb:.0f}GB"}
                    print(f"[config] Using max memory: {max_memory_gb:.0f}GB")
            else:
                kw["torch_dtype"] = torch.bfloat16
                kw["device_map"] = "auto"
                total_memory = torch.cuda.get_device_properties(0).total_memory
                max_memory_gb = total_memory / (1024**3)
                kw["max_memory"] = {0: f"{max_memory_gb:.0f}GB"}
        else:
            kw["torch_dtype"] = torch.float32
            kw["device_map"] = "cpu"

        # Activează Flash Attention prin PyTorch SDPA (doar pt arhitecturi compatibile)
        if self.device == "cuda" and torch.cuda.is_available():
            model_lower = self.model_id.lower()
            sdpa_compatible = any(
                name in model_lower
                for name in ["qwen", "llama", "gemma", "mistral", "phi", "falcon"]
            )
            if sdpa_compatible:
                try:
                    kw["attn_implementation"] = "sdpa"
                    print("[config] Using PyTorch SDPA (FlashAttention built-in)")
                except Exception:
                    print("[config] SDPA not available, using eager attention")
                    kw["attn_implementation"] = "eager"
            else:
                print(f"[config] Model may not support SDPA, using eager attention")
                kw["attn_implementation"] = "eager"

        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Tokenizer loaded")

        # Load model with retry logic
        print("Loading model weights (this may take a few minutes)...")
        try:
            # Încerci fără max_memory dacă e problemă
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    **kw,
                )
            except (ValueError, TypeError) as e:
                if "max_memory" in str(e) or "size" in str(e):
                    print(f"[warning] max_memory issue: {e}")
                    print("[warning] Retrying without max_memory limit...")
                    kw.pop("max_memory", None)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_id,
                        trust_remote_code=True,
                        **kw,
                    )
                else:
                    raise
            
            # Forțează modelul să fie pe GPU și elimină meta device
            if self.device == "cuda" and torch.cuda.is_available():
                print("Ensuring all weights are on GPU...")
                self.model = self.model.to("cuda")
                torch.cuda.synchronize()
                
                # Verifică câte ponderi sunt încărcate
                total_params = sum(p.numel() for p in self.model.parameters())
                params_on_cuda = sum(p.numel() for p in self.model.parameters() if p.is_cuda)
                print(f"Parameters on CUDA: {params_on_cuda:,} / {total_params:,} ({100*params_on_cuda/total_params:.1f}%)")
                
                if params_on_cuda < total_params:
                    print("[warning] Some parameters still on meta/cpu, forcing full load...")
                    for name, param in self.model.named_parameters():
                        if not param.is_cuda:
                            param.data = param.data.to("cuda")
                    torch.cuda.synchronize()
                    
        except Exception as e:
            err_str = str(e)
            
            # If quantization failed, retry without it
            if self.quantization in ("4bit", "8bit") and (
                "bitsandbytes" in err_str.lower() or "cuda" in err_str.lower()
                or "libnvjitlink" in err_str.lower()
            ):
                print(f"Quantization failed: {e}")
                print("Retrying without quantization...")
                kw.pop("quantization_config", None)
                kw.pop("max_memory", None)
                kw["torch_dtype"] = torch.bfloat16
                kw["device_map"] = "auto"
                self.quantization = "none"
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    **kw,
                )
                if self.device == "cuda":
                    self.model = self.model.to("cuda")
                    
            # If model_type error, try forcing Qwen2 type
            elif "Unrecognized model" in err_str and "model_type" in err_str:
                print(f"Model config issue: {e}")
                print("Attempting to force Qwen2 model type...")
                
                if "config" not in kw or not kw["config"]:
                    from transformers import Qwen2Config
                    config = Qwen2Config.from_pretrained(self.model_id, trust_remote_code=True)
                    kw["config"] = config
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    trust_remote_code=True,
                    **kw,
                )
                if self.device == "cuda":
                    self.model = self.model.to("cuda")
            else:
                raise

        self.model.eval()
        self._loaded = True
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")
        
        print("Model loaded successfully and ready for inference")

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("Model unloaded, memory freed")

    def generate(self, messages, max_tokens=4096, temperature=0.2, top_p=0.95):
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        if self.model is None:
            raise RuntimeError("Model is None despite being loaded")

        print("Generating response...", end=" ", flush=True)
        
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=32768
        )
        
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        print(f"input_ids shape: {inputs['input_ids'].shape}, device: {inputs['input_ids'].device}")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        print("done")
        return response.strip()

    @property
    def is_loaded(self):
        return self._loaded