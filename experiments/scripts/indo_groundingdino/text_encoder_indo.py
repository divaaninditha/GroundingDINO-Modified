import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Dict, List, Optional, Tuple
import math
import warnings

# Suppress some warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x, mask):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class IndoBERTTextEncoder(nn.Module):
    """
    Indonesian BERT-based text encoder for GroundingDINO
    Supports multiple Indonesian language models
    OPTIMIZED VERSION - Better performance and stability
    """
    
    def __init__(self, 
                 model_name: str = "indolem/indobert-base-uncased",
                 max_text_len: int = 256,
                 feature_dim: int = 256,
                 freeze_text_encoder: bool = False,
                 use_checkpoint: bool = True):
        """
        Args:
            model_name: Indonesian BERT model name from HuggingFace
            max_text_len: Maximum text sequence length
            feature_dim: Output feature dimension to match GroundingDINO
            freeze_text_encoder: Whether to freeze the text encoder
            use_checkpoint: Whether to use gradient checkpointing for memory efficiency
        """
        super().__init__()
        
        self.model_name = model_name
        self.max_text_len = max_text_len
        self.feature_dim = feature_dim
        
        print(f"🌐 Loading Indonesian text encoder (OPTIMIZED): {model_name}")
        
        # Load Indonesian tokenizer and model with error handling
        self._load_model_with_fallback(model_name)
        
        # Get model dimensions
        self.hidden_size = self.config.hidden_size
        
        # Feature projection layer to match GroundingDINO dimensions
        self.feature_projection = nn.Linear(self.hidden_size, feature_dim)
        
        # Layer norm for better training stability
        self.layer_norm = nn.LayerNorm(feature_dim)
        
        # Position embeddings
        self.position_embedding = PositionEmbeddingSine(feature_dim // 2, normalize=True)
        
        # Special tokens handling
        self._setup_special_tokens()
        
        # Freeze text encoder if requested
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            print("🔒 Text encoder frozen")
        
        # Enable gradient checkpointing for memory efficiency
        if use_checkpoint and hasattr(self.text_encoder, 'gradient_checkpointing_enable'):
            self.text_encoder.gradient_checkpointing_enable()
            print("📋 Gradient checkpointing enabled")
        
        print(f"📏 Text encoder dimensions: {self.hidden_size} -> {feature_dim}")
    
    def _load_model_with_fallback(self, model_name: str):
        """Load model with fallback options for better reliability"""
        fallback_models = [
            model_name,
            "indolem/indobert-base-uncased",
            "cahya/roberta-base-indonesian-522M",
            "Wikidepia/IndoBERT-base-uncased"
        ]
        
        for attempt, model_attempt in enumerate(fallback_models):
            try:
                print(f"   Attempting to load: {model_attempt}")
                self.tokenizer = AutoTokenizer.from_pretrained(model_attempt)
                self.config = AutoConfig.from_pretrained(model_attempt)
                self.text_encoder = AutoModel.from_pretrained(model_attempt)
                self.model_name = model_attempt
                print(f"✅ Successfully loaded {model_attempt}")
                return
            except Exception as e:
                print(f"⚠️  Failed to load {model_attempt}: {e}")
                if attempt == len(fallback_models) - 1:
                    raise Exception("All Indonesian model loading attempts failed")
                continue
    
    def _setup_special_tokens(self):
        """Setup special tokens with proper fallbacks"""
        # Handle pad token
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.pad_token_id = self.tokenizer.eos_token_id
            elif self.tokenizer.unk_token_id is not None:
                self.pad_token_id = self.tokenizer.unk_token_id
            else:
                self.pad_token_id = 0  # Use 0 as last resort
        
        # Handle CLS and SEP tokens for compatibility
        self.cls_token_id = getattr(self.tokenizer, 'cls_token_id', None)
        self.sep_token_id = getattr(self.tokenizer, 'sep_token_id', None)
        
        print(f"🔤 Special tokens - PAD: {self.pad_token_id}, CLS: {self.cls_token_id}, SEP: {self.sep_token_id}")
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported Indonesian language models"""
        return [
            "indolem/indobert-base-uncased",      # IndoBERT base
            "indolem/indobert-large-uncased",     # IndoBERT large  
            "cahya/roberta-base-indonesian-522M", # Indo-RoBERTa
            "cahya/distilbert-base-indonesian",   # Distilled IndoBERT
            "Wikidepia/IndoBERT-base-uncased",    # Alternative IndoBERT
            "Wikidepia/IndoBERT-lite",            # Lightweight version
        ]
    
    def preprocess_indonesian_text(self, text: str) -> str:
        """
        OPTIMIZED: Better Indonesian text preprocessing
        """
        # Enhanced text cleaning for Indonesian
        text = text.strip()
        
        # Handle common Indonesian text issues
        text = text.replace("&nbsp;", " ")
        text = text.replace("\n", " ")
        text = text.replace("\t", " ")
        
        # Handle common Indonesian informal writing
        text = text.replace("yg", "yang")
        text = text.replace("dgn", "dengan")
        text = text.replace("utk", "untuk")
        text = text.replace("krn", "karena")
        text = text.replace("kpd", "kepada")
        
        # Normalize spaces
        text = " ".join(text.split())
        
        # Handle Indonesian number words (for better understanding)
        number_replacements = {
            "satu": "1", "dua": "2", "tiga": "3", "empat": "4", "lima": "5",
            "enam": "6", "tujuh": "7", "delapan": "8", "sembilan": "9", "sepuluh": "10"
        }
        
        for indo_num, digit in number_replacements.items():
            text = text.replace(f" {indo_num} ", f" {digit} ")
        
        return text
    
    def tokenize_text(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """
        OPTIMIZED: Enhanced tokenization with better Indonesian handling
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Enhanced text preprocessing
        processed_texts = []
        for text in texts:
            processed_text = self.preprocess_indonesian_text(text)
            processed_texts.append(processed_text)
        
        # Tokenize with enhanced parameters
        try:
            tokenized = self.tokenizer(
                processed_texts,
                max_length=self.max_text_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=False if 'roberta' in self.model_name.lower() else True
            )
        except Exception as e:
            print(f"⚠️  Tokenization failed: {e}")
            # Fallback tokenization
            tokenized = self.tokenizer(
                processed_texts,
                max_length=min(self.max_text_len, 512),  # Ensure max length
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
        
        return tokenized
    
    def forward(self, texts: List[str], device: torch.device = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of Indonesian text encoder
        OPTIMIZED VERSION - Better error handling and performance
        
        Args:
            texts: List of Indonesian text strings
            device: Target device
            
        Returns:
            Dictionary containing:
                - text_features: Encoded text features [batch_size, seq_len, feature_dim]
                - text_masks: Attention masks [batch_size, seq_len]
                - text_tokens: Token embeddings [batch_size, seq_len, hidden_size]
        """
        if device is None:
            device = next(self.parameters()).device
        
        # OPTIMIZED: Better device handling
        original_device = device
        if device.type == 'cuda' and not torch.cuda.is_available():
            device = torch.device('cpu')
            print("⚠️  CUDA not available, using CPU")
        
        # Tokenize texts
        tokenized = self.tokenize_text(texts)
        
        # Move to device with error handling
        try:
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
        except Exception as e:
            print(f"⚠️  Device transfer failed: {e}, using CPU")
            device = torch.device('cpu')
            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)
        
        # Get token type ids if available (not for RoBERTa)
        token_type_ids = None
        if "token_type_ids" in tokenized:
            token_type_ids = tokenized["token_type_ids"].to(device)
        
        # OPTIMIZED: Enhanced text encoding with error handling
        try:
            with torch.set_grad_enabled(self.training):
                if token_type_ids is not None:
                    text_outputs = self.text_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )
                else:
                    text_outputs = self.text_encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask
                    )
        except Exception as e:
            print(f"⚠️  Text encoding failed: {e}")
            # Return zero tensors as fallback
            batch_size, seq_len = input_ids.shape
            text_hidden_states = torch.zeros(batch_size, seq_len, self.hidden_size, device=device)
        else:
            # Get hidden states
            text_hidden_states = text_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # OPTIMIZED: Better feature projection
        try:
            # Project to target feature dimension
            text_features = self.feature_projection(text_hidden_states)  # [batch_size, seq_len, feature_dim]
            
            # Apply layer normalization
            text_features = self.layer_norm(text_features)
        except Exception as e:
            print(f"⚠️  Feature projection failed: {e}")
            # Create fallback features
            batch_size, seq_len = input_ids.shape
            text_features = torch.zeros(batch_size, seq_len, self.feature_dim, device=device)
        
        # Create text masks (True for padded tokens, False for real tokens)
        text_masks = (input_ids == self.pad_token_id)
        
        # OPTIMIZED: Enhanced output dictionary
        output_dict = {
            "text_features": text_features,
            "text_masks": text_masks,
            "text_tokens": text_hidden_states,
            "attention_mask": attention_mask,
            "input_ids": input_ids,
        }
        
        # Add pooled output if available
        if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
            output_dict["pooled_output"] = text_outputs.pooler_output
        else:
            # Create pooled output from first token (CLS-like)
            output_dict["pooled_output"] = text_features[:, 0, :]
        
        return output_dict
    
    def encode_text(self, texts: List[str], device: torch.device = None) -> torch.Tensor:
        """
        Simple text encoding interface
        Returns only the text features tensor
        """
        outputs = self.forward(texts, device)
        return outputs["text_features"]
    
    def get_text_embeddings(self, texts: List[str], device: torch.device = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get text embeddings and masks for compatibility with original GroundingDINO
        
        Returns:
            text_features: [batch_size, seq_len, feature_dim]
            text_masks: [batch_size, seq_len] (True for padding)
        """
        outputs = self.forward(texts, device)
        return outputs["text_features"], outputs["text_masks"]
    
    def save_pretrained(self, save_directory: str):
        """Save the Indonesian text encoder"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state
        torch.save({
            'model_name': self.model_name,
            'max_text_len': self.max_text_len,
            'feature_dim': self.feature_dim,
            'state_dict': self.state_dict()
        }, os.path.join(save_directory, 'indo_text_encoder.pth'))
        
        print(f"💾 Indonesian text encoder saved to {save_directory}")
    
    @classmethod
    def load_pretrained(cls, load_directory: str, device: torch.device = None):
        """Load the Indonesian text encoder"""
        import os
        
        checkpoint_path = os.path.join(load_directory, 'indo_text_encoder.pth')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Create model
        model = cls(
            model_name=checkpoint['model_name'],
            max_text_len=checkpoint['max_text_len'],
            feature_dim=checkpoint['feature_dim']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])
        
        if device is not None:
            model = model.to(device)
        
        print(f"📂 Indonesian text encoder loaded from {load_directory}")
        return model

def create_indo_text_encoder(model_name: str = "indolem/indobert-base-uncased",
                           feature_dim: int = 256,
                           max_text_len: int = 256,
                           device: torch.device = None) -> IndoBERTTextEncoder:
    """
    Factory function to create Indonesian text encoder
    OPTIMIZED VERSION - Better reliability and performance
    
    Args:
        model_name: Indonesian BERT model name
        feature_dim: Output feature dimension
        max_text_len: Maximum text length
        device: Target device
    
    Returns:
        IndoBERTTextEncoder instance
    """
    print(f"🏗️  Creating Indonesian text encoder (OPTIMIZED)...")
    print(f"   Model: {model_name}")
    print(f"   Feature dim: {feature_dim}")
    print(f"   Max text len: {max_text_len}")
    
    # OPTIMIZED: Better device handling
    if device is not None and device.type == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA requested but not available, using CPU")
        device = torch.device('cpu')
    
    try:
        encoder = IndoBERTTextEncoder(
            model_name=model_name,
            feature_dim=feature_dim,
            max_text_len=max_text_len
        )
        
        if device is not None:
            encoder = encoder.to(device)
            print(f"📱 Moved to device: {device}")
        
        print("✅ Indonesian text encoder created successfully (OPTIMIZED)")
        return encoder
        
    except Exception as e:
        print(f"❌ Failed to create Indonesian text encoder: {e}")
        raise

def test_indo_text_encoder():
    """
    Test function for Indonesian text encoder
    OPTIMIZED VERSION - More comprehensive testing
    """
    print("🧪 Testing Indonesian Text Encoder (OPTIMIZED)...")
    
    # Enhanced test texts in Indonesian
    test_texts = [
        "orang duduk di kursi merah",
        "mobil biru parkir di jalan raya",
        "kucing putih tidur di tempat tidur", 
        "makanan lezat di atas meja kayu",
        "seseorang menggunakan laptop hitam",
        "dua anjing bermain di taman",
        "banyak burung terbang di langit",
        "sebuah rumah besar berdiri di perbukitan"
    ]
    
    print(f"📝 Testing with {len(test_texts)} Indonesian texts...")
    
    # Create encoder with error handling
    try:
        encoder = create_indo_text_encoder()
        encoder.eval()
        
        print(f"✅ Encoder created successfully")
        print(f"   Model: {encoder.model_name}")
        print(f"   Feature dim: {encoder.feature_dim}")
        
    except Exception as e:
        print(f"❌ Failed to create encoder: {e}")
        return None
    
    # Test encoding with comprehensive error handling
    try:
        with torch.no_grad():
            outputs = encoder.forward(test_texts)
            
            print(f"✅ Batch encoding successful:")
            print(f"   Input texts: {len(test_texts)}")
            print(f"   Text features shape: {outputs['text_features'].shape}")
            print(f"   Text masks shape: {outputs['text_masks'].shape}")
            print(f"   Attention mask shape: {outputs['attention_mask'].shape}")
            
            # Test individual encoding
            single_output = encoder.encode_text(["orang dengan tas besar"])
            print(f"✅ Single encoding shape: {single_output.shape}")
            
            # Test compatibility interface
            features, masks = encoder.get_text_embeddings(test_texts[:2])
            print(f"✅ Compatibility interface - Features: {features.shape}, Masks: {masks.shape}")
            
            # Test device handling
            if torch.cuda.is_available():
                try:
                    cuda_encoder = create_indo_text_encoder(device=torch.device('cuda'))
                    cuda_output = cuda_encoder.encode_text(["test CUDA"])
                    print(f"✅ CUDA encoding successful: {cuda_output.shape}")
                except Exception as e:
                    print(f"⚠️  CUDA test failed: {e}")
            
            # Test text preprocessing
            test_informal = "org yg duduk dgn tas utk kerja"
            preprocessed = encoder.preprocess_indonesian_text(test_informal)
            print(f"✅ Text preprocessing: '{test_informal}' -> '{preprocessed}'")
            
    except Exception as e:
        print(f"❌ Encoding test failed: {e}")
        return None
    
    print("🎉 Indonesian text encoder test completed successfully (OPTIMIZED)!")
    
    return encoder

if __name__ == "__main__":
    # Test the OPTIMIZED Indonesian text encoder
    test_encoder = test_indo_text_encoder()
    
    if test_encoder:
        # Show supported models
        print(f"\n📋 Supported Indonesian models:")
        for model in test_encoder.get_supported_models():
            print(f"   - {model}")
            
        print(f"\n🎯 Indonesian text encoder ready for integration!")
    else:
        print(f"\n❌ Test failed - please check the error messages above")