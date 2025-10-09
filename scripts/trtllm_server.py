#!/usr/bin/env python3
"""
Serveur API OpenAI-compatible pour TensorRT-LLM (Qwen 2.5 1.5B)
Compatible avec votre code ASR existant via LLM_ENDPOINT
"""
import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# TensorRT-LLM imports
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner
from transformers import AutoTokenizer


# =============================================================================
# Models Pydantic (format OpenAI)
# =============================================================================

class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.1
    max_tokens: int = 512
    stream: bool = False


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time())}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Dict]


# =============================================================================
# TensorRT-LLM Runner
# =============================================================================

class TRTLLMRunner:
    def __init__(self, engine_dir: str, tokenizer_path: str):
        """
        Initialise le moteur TensorRT-LLM et le tokenizer.

        Args:
            engine_dir: Chemin vers le répertoire du moteur TRT
            tokenizer_path: Chemin ou nom HuggingFace du tokenizer
        """
        print(f"[INFO] Loading TensorRT-LLM engine from: {engine_dir}")
        self.engine_dir = Path(engine_dir)

        if not self.engine_dir.exists():
            raise FileNotFoundError(f"Engine directory not found: {engine_dir}")

        # Charger le tokenizer
        print(f"[INFO] Loading tokenizer: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            padding_side="left"
        )

        # Initialiser le runner TensorRT-LLM
        self.runner = ModelRunner.from_dir(
            engine_dir=str(self.engine_dir),
            rank=0,
            debug_mode=False
        )

        print("[INFO] TensorRT-LLM engine loaded successfully!")

    def generate(
            self,
            prompt: str,
            max_new_tokens: int = 512,
            temperature: float = 0.1,
            top_p: float = 0.9,
            top_k: int = 50,
    ) -> str:
        """
        Génère du texte avec le moteur TensorRT-LLM.

        Args:
            prompt: Prompt d'entrée
            max_new_tokens: Nombre maximum de tokens à générer
            temperature: Température de sampling
            top_p: Nucleus sampling
            top_k: Top-k sampling

        Returns:
            Texte généré
        """
        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_lengths = [input_ids.shape[1]]

        # Génération avec TensorRT-LLM
        outputs = self.runner.generate(
            batch_input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            end_id=self.tokenizer.eos_token_id,
            pad_id=self.tokenizer.pad_token_id,
            num_beams=1,
            output_sequence_lengths=True,
            return_dict=True
        )

        # Decode
        output_ids = outputs['output_ids'][0][0]  # [batch, beam, seq]
        output_text = self.tokenizer.decode(
            output_ids[input_lengths[0]:],  # Skip input tokens
            skip_special_tokens=True
        )

        return output_text.strip()


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(title="TensorRT-LLM API", version="1.0.0")
runner: Optional[TRTLLMRunner] = None


@app.on_event("startup")
async def startup_event():
    """Charge le modèle au démarrage du serveur."""
    global runner
    if runner is None:
        engine_dir = app.state.engine_dir
        tokenizer_path = app.state.tokenizer_path
        runner = TRTLLMRunner(engine_dir, tokenizer_path)


@app.get("/health")
async def health_check():
    """Endpoint de santé."""
    if runner is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model": "qwen2.5-1.5b-instruct"}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Endpoint compatible OpenAI /v1/chat/completions.
    """
    if runner is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Construire le prompt Qwen format
    # Format Qwen 2.5: <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n
    prompt_parts = []
    for msg in request.messages:
        if msg.role == "system":
            prompt_parts.append(f"<|im_start|>system\n{msg.content}<|im_end|>")
        elif msg.role == "user":
            prompt_parts.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
        elif msg.role == "assistant":
            prompt_parts.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")

    prompt_parts.append("<|im_start|>assistant\n")
    prompt = "\n".join(prompt_parts)

    # Génération
    try:
        output = runner.generate(
            prompt=prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
        )

        response = ChatCompletionResponse(
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": output
                },
                "finish_reason": "stop"
            }]
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "TensorRT-LLM API Server",
        "model": "Qwen 2.5 1.5B Instruct",
        "endpoints": {
            "health": "/health",
            "chat": "/v1/chat/completions"
        }
    }


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="TensorRT-LLM API Server")
    parser.add_argument("--engine_dir", type=str, required=True,
                        help="Path to TensorRT engine directory")
    parser.add_argument("--tokenizer", type=str, required=True,
                        help="Tokenizer path or HuggingFace model name")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to bind")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port to bind")
    args = parser.parse_args()

    # Store args in app state
    app.state.engine_dir = args.engine_dir
    app.state.tokenizer_path = args.tokenizer

    print("=" * 60)
    print("TensorRT-LLM API Server")
    print("=" * 60)
    print(f"Engine: {args.engine_dir}")
    print(f"Tokenizer: {args.tokenizer}")
    print(f"Server: http://{args.host}:{args.port}")
    print("=" * 60)

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


if __name__ == "__main__":
    main()