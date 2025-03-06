# estimator.py
import math
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from together import Together
from transformers import LogitsProcessor, PreTrainedTokenizer

# Import analyzers with conditional checks
from .core.analyzer import EntropyAnalyzer
try:
    from .core.analyzer import VLMAnalyzer, EnhancedVLMAnalyzer
    HAS_VLM_ANALYZERS = True
except ImportError:
    HAS_VLM_ANALYZERS = False
    # Create placeholder classes for type checking
    class VLMAnalyzer(EntropyAnalyzer): pass
    class EnhancedVLMAnalyzer(VLMAnalyzer): pass

from .models import TokenInfo, UncertaintyAnalysisResult, UncertaintyMetrics


class UncertaintyLogitsProcessor(LogitsProcessor):
    def __init__(self, estimator):
        self.captured_logits = []
        self.estimator = estimator
        self.input_ids = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.input_ids is None:
            self.input_ids = input_ids
        self.captured_logits.append(scores.detach().clone())
        return scores


class UncertaintyEstimator:
    def __init__(
        self,
        top_k: int = 100,
        analyzer: Optional[EntropyAnalyzer] = None,
        together_api_key: Optional[str] = None,
        together_model: Optional[str] = None,
    ):
        self.analyzer = analyzer
        self.top_k = 5 if together_model else top_k
        self.together_client = Together(api_key=together_api_key) if together_api_key else None
        self.together_model = together_model
        self.is_enhanced_vlm = isinstance(analyzer, EnhancedVLMAnalyzer)

    def get_logits_processor(self) -> LogitsProcessor:
        """Get appropriate logits processor based on model type"""
        return UncertaintyLogitsProcessor(self)

    def _process_logits(self, logits: torch.Tensor, tokenizer: PreTrainedTokenizer) -> List[TokenInfo]:
        """Process HuggingFace logits into TokenInfo objects"""
        probs = torch.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs[0], self.top_k)
        top_logits = logits[0][top_indices]

        token_info = []
        for token_id, logit, prob in zip(top_indices, top_logits, top_probs):
            token = tokenizer.decode(token_id)
            token_info.append(
                TokenInfo(
                    token=token,
                    token_id=token_id.item(),
                    logit=logit.item(),
                    probability=prob.item(),
                )
            )
        return token_info

    def _process_together_logprob(self, logprob: float) -> float:
        """Convert logprob to probability correctly"""
        return float(np.exp(logprob))

    def _generate_with_together(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text using Together AI chat with logprobs"""
        messages = [{"role": "user", "content": prompt}]
        generation_config = {
            "max_tokens": kwargs.get("max_new_tokens", 10),
            "temperature": kwargs.get("temperature", 0.7),
            "logprobs": True,
            "stream": False,
        }

        response = self.together_client.chat.completions.create(
            model=self.together_model, messages=messages, **generation_config
        )

        logprobs_data = response.choices[0].logprobs
        return {
            "text": response.choices[0].message.content,
            "tokens": logprobs_data.tokens,
            "token_logprobs": logprobs_data.token_logprobs,
            "token_ids": logprobs_data.token_ids,
        }

    def analyze_generation(
        self,
        generation_output: Any,
        model: Optional[Any] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        processor: Optional[LogitsProcessor] = None,
        prompt: Optional[str] = None,
        image: Optional[Any] = None,
    ) -> UncertaintyAnalysisResult:
        all_metrics = []
        generated_text = ""
        attention_data = None
        input_query = prompt or ""

        # Check if this is a VLM output and which type of analyzer we're using
        is_vlm = hasattr(generation_output, "attentions") and isinstance(
            self.analyzer, (VLMAnalyzer, EnhancedVLMAnalyzer)
        )

        # Check if we're using VLM analyzers and they're available
        if isinstance(self.analyzer, (VLMAnalyzer, EnhancedVLMAnalyzer)) and not HAS_VLM_ANALYZERS:
            raise ImportError(
                "VLM analyzers are not available. Please install the required dependencies: "
                "pip install 'klarity[vllm]' or pip install vllm"
            )

        if is_vlm:
            if not hasattr(self.analyzer, "patch_size") or self.analyzer.patch_size is None:
                self.analyzer.set_vision_config(model.config.vision_config)

            # Process VLM-specific outputs
            input_length = processor.input_ids.shape[1] if hasattr(processor, "input_ids") else 0
            generated_tokens = generation_output.sequences[0][input_length:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Get individual tokens
            tokens = []
            for i in range(len(generated_tokens)):
                token_text = tokenizer.decode(generated_tokens[i : i + 1], skip_special_tokens=True)
                if token_text.strip():
                    tokens.append(token_text)

            # Process attention maps
            attention_data = self.analyzer.process_attention_maps(generation_output.attentions, tokens)

            # Process token predictions
            if hasattr(generation_output, "scores"):
                for step, logits in enumerate(generation_output.scores):
                    token_info = self._process_logits(logits, tokenizer)

                    metrics = UncertaintyMetrics(
                        raw_entropy=self.analyzer._calculate_raw_entropy(np.array([t.probability for t in token_info])),
                        semantic_entropy=self.analyzer._calculate_semantic_entropy(token_info),
                        token_predictions=token_info,
                    )
                    all_metrics.append(metrics)

            # Generate insight based on analyzer type
            if self.is_enhanced_vlm and HAS_VLM_ANALYZERS:
                # For EnhancedVLMAnalyzer, always use visual analysis
                overall_insight = self.analyzer.generate_overall_insight(
                    metrics_list=all_metrics,
                    input_query=input_query,
                    generated_text=generated_text,
                    attention_data=attention_data,
                    image=image,
                    use_visual_analysis=True,
                )
            else:
                # For regular VLMAnalyzer
                overall_insight = self.analyzer.generate_overall_insight(
                    metrics_list=all_metrics,
                    input_query=input_query,
                    generated_text=generated_text,
                    attention_data=attention_data,
                )

            return UncertaintyAnalysisResult(
                token_metrics=all_metrics, overall_insight=overall_insight, attention_data=attention_data
            )

        else:
            # Handle non-VLM outputs (existing code for VLLM, Together, HuggingFace)
            if hasattr(generation_output, "outputs"):
                # VLLM outputs handling
                generated_text = generation_output.outputs[0].text
                logprobs_data = generation_output.outputs[0].logprobs

                if logprobs_data:
                    for token_data in logprobs_data:
                        logprobs_items = [
                            (token, logprob.logprob, logprob.decoded_token) for token, logprob in token_data.items()
                        ]
                        logprobs_items.sort(key=lambda x: x[1], reverse=True)

                        token_info = [
                            TokenInfo(
                                token=decoded_token,
                                token_id=int(token_id),
                                logit=logprob,
                                probability=math.exp(logprob),
                            )
                            for token_id, logprob, decoded_token in logprobs_items[: self.top_k]
                        ]

                        if token_info:
                            metrics = UncertaintyMetrics(
                                raw_entropy=self.analyzer._calculate_raw_entropy(
                                    np.array([t.probability for t in token_info])
                                )
                                if self.analyzer
                                else 0.0,
                                semantic_entropy=self.analyzer._calculate_semantic_entropy(token_info)
                                if self.analyzer
                                else 0.0,
                                token_predictions=token_info,
                            )
                            all_metrics.append(metrics)

            elif self.together_model:
                # Together API outputs handling
                generated_text = generation_output["text"]
                for step in range(len(generation_output["tokens"])):
                    prob = self._process_together_logprob(generation_output["token_logprobs"][step])
                    token_info = [
                        TokenInfo(
                            token=generation_output["tokens"][step],
                            token_id=generation_output["token_ids"][step],
                            logit=generation_output["token_logprobs"][step],
                            probability=prob,
                        )
                    ]
                    metrics = UncertaintyMetrics(
                        raw_entropy=1 - prob,
                        semantic_entropy=0.0,
                        token_predictions=token_info,
                    )
                    all_metrics.append(metrics)

            else:
                # Handle HuggingFace outputs
                if not tokenizer or not processor:
                    raise ValueError("Tokenizer and processor required for HuggingFace models")

                generated_text = tokenizer.decode(generation_output.sequences[0], skip_special_tokens=True)
                for step, logits in enumerate(processor.captured_logits):
                    token_info = self._process_logits(logits, tokenizer)
                    metrics = UncertaintyMetrics(
                        raw_entropy=self.analyzer._calculate_raw_entropy(np.array([t.probability for t in token_info])),
                        semantic_entropy=self.analyzer._calculate_semantic_entropy(token_info),
                        token_predictions=token_info,
                    )
                    all_metrics.append(metrics)

            # Generate insight for non-VLM case
            overall_insight = self.analyzer.generate_overall_insight(
                all_metrics, input_query=input_query, generated_text=generated_text
            )

        return UncertaintyAnalysisResult(
            token_metrics=all_metrics,
            overall_insight=overall_insight,
            attention_data=attention_data if is_vlm else None,
        )
