"""Tokenizer for Canary MLX models."""

import re
from functools import cached_property
from pathlib import Path
from typing import Union

try:
    import sentencepiece as spm
except ImportError:
    spm = None


class CanaryTokenizer:
    """
    Tokenizer for Canary models using SentencePiece.
    
    The Canary tokenizer uses a unified SentencePiece vocabulary that includes
    both regular tokens and special control tokens.
    """
    
    # Special token constants
    BOS = "<|startoftranscript|>"
    EOS = "<|endoftext|>"
    PAD = "<pad>"
    NOSPEECH = "<|nospeech|>"
    PNC = "<|pnc|>"
    NOPNC = "<|nopnc|>"
    CONTEXT_START = "<|startofcontext|>"
    
    def __init__(self, sp: "spm.SentencePieceProcessor", special_vocab: dict, regular_vocab: dict):
        self.sp = sp
        self._special_vocab = special_vocab
        self._regular_vocab = regular_vocab
        self._full_vocab = {**special_vocab, **regular_vocab}
        self._id_to_token = {v: k for k, v in self._full_vocab.items()}
        
        self.special_tokens = {
            token: tid for token, tid in special_vocab.items()
        }
    
    @staticmethod
    def from_file(model_path: Union[str, Path]) -> "CanaryTokenizer":
        """
        Load tokenizer from a SentencePiece model file.
        
        Args:
            model_path: Path to the .model file.
            
        Returns:
            CanaryTokenizer instance.
        """
        if spm is None:
            raise ImportError("sentencepiece is required. Install with: pip install sentencepiece")
        
        sp = spm.SentencePieceProcessor()
        sp.Load(str(model_path))
        
        vocab = {sp.id_to_piece(i): i for i in range(sp.vocab_size())}
        
        # Identify special tokens by pattern
        special_pattern = re.compile(r'^<\|.*\|>$|^<pad>$|^<unk>$')
        special_vocab = {k: v for k, v in vocab.items() if special_pattern.match(k)}
        regular_vocab = {k: v for k, v in vocab.items() if not special_pattern.match(k)}
        
        return CanaryTokenizer(sp, special_vocab, regular_vocab)
    
    def encode(self, text: str, lang_id: str = None) -> list[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Text to encode. Can contain special tokens like <|en|>.
            lang_id: Language ID (optional). Use "spl_tokens" to encode only special tokens.
            
        Returns:
            List of token IDs.
        """
        if lang_id == "spl_tokens":
            tokens = re.findall(r"<\|[^|]+\|>", text)
            return [self.special_tokens[t] for t in tokens if t in self.special_tokens]
        
        return self.sp.encode(text)
    
    def decode(self, token_ids: list[int], strip: bool = True) -> str:
        """
        Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs to decode.
            strip: If True, strip leading/trailing whitespace.
            
        Returns:
            Decoded text.
        """
        regular_ids = [tid for tid in token_ids if tid not in self._special_vocab.values()]
        
        if not regular_ids:
            return ""
        
        if len(regular_ids) == 1 and not strip:
            piece = self.sp.id_to_piece(regular_ids[0])
            return piece.replace("▁", " ")
        
        text = self.sp.decode(regular_ids)
        text = text.replace("▁", " ")
        return text.strip() if strip else text
    
    def decode_with_special(self, token_ids: list[int]) -> str:
        """Decode token IDs including special tokens."""
        pieces = []
        for tid in token_ids:
            if tid in self._id_to_token:
                pieces.append(self._id_to_token[tid])
            else:
                pieces.append(f"[UNK:{tid}]")
        return "".join(pieces).replace("▁", " ")
    
    @cached_property
    def eos_id(self) -> int:
        return self.special_tokens[self.EOS]

    @cached_property
    def bos_id(self) -> int:
        return self.special_tokens[self.BOS]

    @cached_property
    def nospeech_id(self) -> int:
        return self.special_tokens[self.NOSPEECH]

    @cached_property
    def pad_id(self) -> int:
        return self.special_tokens[self.PAD]

