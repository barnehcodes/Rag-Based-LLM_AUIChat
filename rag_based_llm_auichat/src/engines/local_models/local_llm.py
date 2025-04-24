"""
Local LLM module for AUIChat
Implements an interface compatible with llama_index LLM interface
"""
from .model_loader import create_local_llm_handler
import time
from llama_index.core.llms import (
    CompletionResponse, 
    CompletionResponseGen, 
    LLMMetadata,
    LLM,
    ChatMessage, 
    MessageRole, 
    ChatResponse, 
    ChatResponseGen
)
from typing import Any, Dict, Optional, Sequence, Union
from pydantic import Field, PrivateAttr

class LocalLLM(LLM):
    """
    A wrapper class for local LLMs that implements the llama_index LLM interface
    """
    # Define class fields properly for Pydantic
    model_name: str = Field(default="SmolLM-360M")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=512)
    
    # Use private attribute for the pipeline
    _pipeline: Any = PrivateAttr(default=None)
    
    def __init__(self, **kwargs):
        """Initialize the local LLM handler"""
        super().__init__(**kwargs)
        self._pipeline = create_local_llm_handler()
        
    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            model_name=self.model_name,
            max_input_tokens=4096,
            max_output_tokens=self.max_tokens,
            is_chat_model=True,  # Mark as chat-capable
            is_function_calling_model=False,
        )
    
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """
        Complete the prompt with the local model
        
        Args:
            prompt (str): The input prompt
            **kwargs: Additional arguments for generation
        
        Returns:
            CompletionResponse: The generated text response
        """
        # Extract generation parameters
        max_new_tokens = kwargs.get('max_tokens', self.max_tokens)
        temperature = kwargs.get('temperature', self.temperature)
        
        # Print info about the request
        start_time = time.time()
        print(f"Generating response with local {self.model_name} model...")
        print(f"Prompt length: {len(prompt)} characters")
        
        # Run the local model
        results = self._pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            return_full_text=False,
            pad_token_id=50256,
        )
        
        # Extract the response text
        response = results[0]['generated_text']
        
        # Log generation time
        inference_time = time.time() - start_time
        print(f"✓ Response generated in {inference_time:.2f}s ({len(response)} characters)")
        
        return CompletionResponse(text=response)
    
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """
        Chat implementation by converting messages to a single prompt then using completion
        
        Args:
            messages: A sequence of ChatMessage objects
            **kwargs: Additional arguments for generation
            
        Returns:
            ChatResponse: The model's response
        """
        # Convert messages to a single prompt
        prompt = self._messages_to_prompt(messages)
        
        # Get completion response using the standard completion method
        completion_response = self.complete(prompt, **kwargs)
        
        # Convert completion response to chat response
        return ChatResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=completion_response.text
            ),
            raw=completion_response.raw,
            delta=None,
        )
    
    def _messages_to_prompt(self, messages: Sequence[ChatMessage]) -> str:
        """
        Convert a sequence of ChatMessage objects to a single prompt string,
        prioritizing the system prompt and the latest user message to fit
        within context limits.
        
        Args:
            messages: A sequence of ChatMessage objects
            
        Returns:
            str: A prompt string formatted for the model
        """
        prompt_parts = []
        system_prompt = "SYSTEM: You are an AI assistant for Al Akhawayn University in Ifrane. Please answer the following questions accurately and helpfully based on the provided context."
        last_user_message = None

        # Find the last user message
        for msg in reversed(messages):
            if msg.role == MessageRole.USER:
                last_user_message = msg
                break
        
        # Add the system prompt
        prompt_parts.append(system_prompt)
        
        # Add the last user message if found
        if last_user_message:
            prompt_parts.append(f"USER: {last_user_message.content}")
        else:
            # Fallback if no user message (shouldn't happen with current API structure)
            prompt_parts.append("USER: Please provide information.")
            
        # Add the final assistant prompt
        prompt_parts.append("ASSISTANT:")
            
        final_prompt = "\n".join(prompt_parts)
        
        # Optional: Add a check/truncation based on token count if needed, 
        # but limiting to system + last user message should generally work for SmolLM-360M
        # max_prompt_tokens = 1800 # Leave some room for generation
        # tokenized_prompt = self._pipeline.tokenizer.encode(final_prompt)
        # if len(tokenized_prompt) > max_prompt_tokens:
        #     truncated_tokens = tokenized_prompt[:max_prompt_tokens]
        #     final_prompt = self._pipeline.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        #     # Ensure it ends correctly
        #     if not final_prompt.endswith("ASSISTANT:"):
        #         final_prompt += "\nASSISTANT:"
        #     print(f"Warning: Prompt truncated to {len(final_prompt)} characters to fit context window.")

        return final_prompt
    
    def stream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Not implemented for this model."""
        # For now, we don't support streaming, so we just wrap the completion
        completion_response = self.complete(prompt, **kwargs)
        
        def gen():
            yield completion_response
            
        return gen()
        
    def stream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        """Not implemented for this model."""
        # For now, we don't support streaming, so we just wrap the chat
        chat_response = self.chat(messages, **kwargs)
        
        def gen():
            yield chat_response
            
        return gen()
    
    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Async completion not implemented."""
        # Just call the sync version for now
        return self.complete(prompt, **kwargs)
    
    async def astream_complete(self, prompt: str, **kwargs: Any) -> CompletionResponseGen:
        """Async streaming not implemented."""
        # For now, just call the sync version
        return self.stream_complete(prompt, **kwargs)
        
    async def achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        """Async chat implementation."""
        # Just call the sync version for now
        return self.chat(messages, **kwargs)
    
    async def astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponseGen:
        """Async chat streaming implementation."""
        # Just call the sync version for now
        return self.stream_chat(messages, **kwargs)