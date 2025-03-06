# core/analyzer.py
import json
import os
import re
import tempfile
import traceback
from collections import defaultdict
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
# Make xgrammar optional
try:
    import xgrammar as xgr
    HAS_XGRAMMAR = True
except ImportError:
    HAS_XGRAMMAR = False
from PIL import Image
from pydantic import BaseModel
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

from klarity.core.schemas.insight_schemas import InsightAnalysisResponseModel
from klarity.core.schemas.reasoning_analysis_schemas import (
    ReasoningStepAnalysisResponseModel,
    ReasoningStepIdentificationResponseModel,
)
from klarity.core.schemas.vlm_analysis_schemas import EnhancedVLMAnalysisResponseModel, VLMAnalysisResponseModel
from klarity.core.system_prompts import *
from klarity.core.together_wrapper import TogetherModelWrapper

from ..models import AttentionData, TokenInfo, UncertaintyAnalysisRequest, UncertaintyMetrics


class EntropyAnalyzer:
    def __init__(
        self,
        min_token_prob: float = 0.01,
        insight_model: Optional[Any] = None,
        insight_tokenizer: Optional[Any] = None,
        insight_api_key: Optional[str] = None,
        insight_prompt_template: Optional[str] = None,
        insight_response_model: BaseModel = InsightAnalysisResponseModel,
    ):
        self.embedding_model = SentenceTransformer("minishlab/potion-base-8M")
        self.min_token_prob = min_token_prob

        # Initialize Together AI model if specified
        self.together_model = None
        self.insight_model = None
        self.insight_tokenizer = None

        if isinstance(insight_model, str) and insight_model.startswith("together:"):
            model_name = insight_model.replace("together:", "")
            self.together_model = TogetherModelWrapper(model_name=model_name, api_key=insight_api_key)
        else:
            self.insight_model = insight_model
            self.insight_tokenizer = insight_tokenizer

        self.insight_prompt_template = insight_prompt_template or INSIGHT_PROMPT_TEMPLATE
        self.insight_response_model = insight_response_model

    def _calculate_raw_entropy(self, probabilities: np.ndarray) -> float:
        """Calculate raw entropy of probability distribution"""
        max_entropy = np.log(len(probabilities))
        raw_entropy = entropy(probabilities)
        return raw_entropy / max_entropy if max_entropy != 0 else 0.0

    def _calculate_semantic_entropy(self, token_info: List[TokenInfo]) -> float:
        """Calculate semantic entropy based on token predictions"""
        if len(token_info) < 2:
            return 0.0

        tokens = [t.token for t in token_info]
        embeddings = self.embedding_model.encode(tokens)
        similarity_matrix = cosine_similarity(embeddings)
        semantic_groups = self._group_similar_tokens(similarity_matrix, token_info)
        group_probs = self._calculate_group_probabilities(semantic_groups, token_info)

        if len(group_probs) > 1:
            return entropy(list(group_probs.values())) / np.log(len(group_probs))
        return 0.0

    def _group_similar_tokens(
        self,
        similarity_matrix: np.ndarray,
        token_info: List[TokenInfo],
        threshold: float = 0.8,
    ) -> Dict[int, List[int]]:
        """Group tokens with similar semantic meanings"""
        groups = defaultdict(list)
        group_id = 0

        for i in range(len(token_info)):
            assigned = False
            for gid in groups:
                if any(similarity_matrix[i][j] > threshold for j in groups[gid]):
                    groups[gid].append(i)
                    assigned = True
                    break
            if not assigned:
                groups[group_id] = [i]
                group_id += 1

        return groups

    def _calculate_group_probabilities(
        self, semantic_groups: Dict[int, List[int]], token_info: List[TokenInfo]
    ) -> Dict[int, float]:
        """Calculate probability mass for each semantic group"""
        group_probs = defaultdict(float)
        total_prob = 0.0

        for gid, indices in semantic_groups.items():
            group_prob = sum(token_info[i].probability for i in indices)
            group_probs[gid] = group_prob
            total_prob += group_prob

        if total_prob > 0:
            for gid in group_probs:
                group_probs[gid] /= total_prob

        return group_probs

    def generate_overall_insight(
        self,
        metrics_list: List[UncertaintyMetrics],
        input_query: Optional[str] = "",
        generated_text: Optional[str] = "",
    ) -> Optional[str]:
        """Generate overall insight for all collected metrics"""
        if not self.together_model and not (self.insight_model and self.insight_tokenizer):
            return None

        # Format metrics for the prompt
        detailed_metrics = []
        for idx, metrics in enumerate(metrics_list):
            top_predictions = [f"{t.token} ({t.probability:.3f})" for t in metrics.token_predictions[:3]]

            step_metrics = (
                f"Step {idx}:\n"
                f"- Raw Entropy: {metrics.raw_entropy:.4f}\n"
                f"- Semantic Entropy: {metrics.semantic_entropy:.4f}\n"
                f"- Top 3 Predictions: {' | '.join(top_predictions)}"
            )
            detailed_metrics.append(step_metrics)

        all_metrics = "\n\n".join(detailed_metrics)
        prompt = self.insight_prompt_template.format(
            detailed_metrics=all_metrics,
            input_query=input_query,
            generated_text=generated_text,
        )

        if self.together_model:
            return self.together_model.generate_insight(prompt, response_model=self.insight_response_model)

        # If instance is vllm, use guided decoding
        elif isinstance(self.insight_model, LLM):
            guided_decoding_params = GuidedDecodingParams(
                json=self.insight_response_model.model_json_schema(),
            )
            sampling_params = SamplingParams(
                guided_decoding=guided_decoding_params,
                max_tokens=400,
                temperature=0.7,
                top_p=0.9,
            )

            response = self.insight_model.generate(
                prompt,
                sampling_params=sampling_params,
            )
            return response[0].outputs[0].text

        # Assume HuggingFace model
        else:
            inputs = self.insight_tokenizer(prompt, return_tensors="pt").to(self.insight_model.device)
            
            # Use xgrammar to enforce structured outputs if available
            if HAS_XGRAMMAR and self.insight_tokenizer and self.insight_response_model:
                try:
                    tokenizer_info = xgr.TokenizerInfo.from_huggingface(self.insight_tokenizer)
                    grammar_compiler = xgr.GrammarCompiler(tokenizer_info)
                    compiled_grammar = grammar_compiler.compile_json_schema(self.insight_response_model)
                    xgr_logits_processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)
                    
                    # Generate with grammar constraints
                    outputs = self.insight_model.generate(
                        **inputs,
                        max_new_tokens=400,
                        temperature=0.7,
                        top_p=0.9,
                        logits_processor=[xgr_logits_processor],
                    )
                except Exception as e:
                    print(f"Error using xgrammar: {e}. Falling back to standard generation.")
                    outputs = self.insight_model.generate(
                        **inputs,
                        max_new_tokens=400,
                        temperature=0.7,
                        top_p=0.9,
                    )
            else:
                # Standard generation without grammar constraints
                outputs = self.insight_model.generate(
                    **inputs,
                    max_new_tokens=400,
                    temperature=0.7,
                    top_p=0.9,
                )
            return self.insight_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def analyze(self, request: UncertaintyAnalysisRequest) -> UncertaintyMetrics:
        """Analyze uncertainty for a single generation step"""
        probabilities = np.array([t.probability for t in request.token_info])
        raw_entropy = self._calculate_raw_entropy(probabilities)
        semantic_entropy = self._calculate_semantic_entropy(request.token_info)

        metrics = UncertaintyMetrics(
            raw_entropy=float(raw_entropy),
            semantic_entropy=float(semantic_entropy),
            token_predictions=request.token_info,
        )

        return metrics


class ReasoningAnalyzer(EntropyAnalyzer):
    def __init__(
        self,
        reasoning_start_token: str = "<think>",
        reasoning_end_token: str = "</think>",
        reasoning_identification_response_model: BaseModel = ReasoningStepIdentificationResponseModel,
        reasoning_step_analysis_response_model: BaseModel = ReasoningStepAnalysisResponseModel,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.reasoning_start_token = reasoning_start_token
        self.reasoning_end_token = reasoning_end_token

        # Template to identify reasoning steps
        self.reasoning_identification_template = REASONING_STEP_IDENTIFICATION_PROMPT_TEMPLATE

        self.step_analysis_template = REASONING_STEP_ANALYSIS_PROMPT_TEMPLATE
        self.reasoning_step_analysis_response_model = reasoning_step_analysis_response_model
        self.reasoning_step_identification_response_model = reasoning_step_identification_response_model

    def identify_reasoning_steps(self, text: str) -> List[Dict]:
        """Use the insight model to identify reasoning steps"""
        try:
            prompt = self.reasoning_identification_template.format(
                start_token=self.reasoning_start_token,
                end_token=self.reasoning_end_token,
                text=text,
            )

            if self.together_model:
                response = self.together_model.generate_insight(
                    prompt, self.reasoning_step_identification_response_model
                )
                print("\nDEBUG - Raw response:")
                print(response)

                # Clean up the response by removing markdown code blocks
                cleaned_response = response.replace("```json", "").replace("```", "").strip()

                try:
                    parsed = json.loads(cleaned_response)
                    print("\nDEBUG - Parsed JSON:")
                    print(json.dumps(parsed, indent=2))
                    return parsed.get("reasoning_steps", [])
                except json.JSONDecodeError as e:
                    print(f"\nDEBUG - JSON parsing error: {e}")
                    print("Cleaned response:", cleaned_response)
                    return []

        except Exception as e:
            print(f"Error in identify_reasoning_steps: {str(e)}")
            return []

    def analyze_reasoning_step(
        self,
        step_info: Dict,
        metrics_list: List[UncertaintyMetrics],
        input_query: str,
        total_steps: int,
    ) -> Dict:
        try:
            start_idx = step_info["position"][0]
            end_idx = step_info["position"][1]
            step_metrics = self._get_metrics_for_range(metrics_list, start_idx, end_idx)

            detailed_metrics = self._format_metrics(step_metrics)

            prompt = self.step_analysis_template.format(
                reasoning_content=step_info["content"],
                input_query=input_query,
                step_number=step_info.get("step_number", 1),
                total_steps=total_steps,
                detailed_metrics=detailed_metrics,
            )

            if self.together_model:
                response = self.together_model.generate_insight(prompt, self.reasoning_step_analysis_response_model)
                print(f"\nDEBUG - Raw response from analysis: {response}")

                # Find JSON content
                start = response.find("{")
                end = response.rfind("}") + 1

                if start >= 0 and end > start:
                    json_str = response[start:end]
                    # Remove any potential line breaks or comments in the JSON
                    json_str = re.sub(r"#.*$", "", json_str, flags=re.MULTILINE)
                    json_str = json_str.replace("\n", "").replace("\\n", "")

                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        print(f"JSON parsing error: {e}")
                        print(f"Attempted to parse: {json_str}")
                        # Return a default structure
                        return {
                            "training_insights": {
                                "step_quality": {
                                    "coherence": "0.5",
                                    "relevance": "0.5",
                                    "confidence": "0.5",
                                },
                                "improvement_targets": [],
                                "tokens_of_interest": [],
                            }
                        }

                return {"error": "No JSON found in response"}

        except Exception as e:
            print(f"Error in analyze_reasoning_step: {str(e)}")
            traceback.print_exc()  # Add this for more detailed error info
            return {"error": f"Analysis failed: {str(e)}"}

    def _get_metrics_for_range(
        self, metrics_list: List[UncertaintyMetrics], start_idx: int, end_idx: int
    ) -> List[UncertaintyMetrics]:
        """Extract metrics for a specific token range"""
        return metrics_list[start_idx:end_idx]

    def _format_metrics(self, metrics: List[UncertaintyMetrics]) -> str:
        """Format metrics for the prompt"""
        formatted = []
        for idx, metric in enumerate(metrics):
            top_predictions = [f"{t.token} ({t.probability:.3f})" for t in metric.token_predictions[:3]]

            metric_text = (
                f"Token {idx}:\n"
                f"- Raw Entropy: {metric.raw_entropy:.4f}\n"
                f"- Semantic Entropy: {metric.semantic_entropy:.4f}\n"
                f"- Top Predictions: {' | '.join(top_predictions)}"
            )
            formatted.append(metric_text)

        return "\n\n".join(formatted)

    def generate_overall_insight(
        self,
        metrics_list: List[UncertaintyMetrics],
        input_query: Optional[str] = "",
        generated_text: Optional[str] = "",
    ) -> Optional[Dict]:
        """Generate comprehensive analysis of all reasoning steps"""
        if not (self.together_model or (self.insight_model and self.insight_tokenizer)):
            return None

        # Identify reasoning steps
        reasoning_steps = self.identify_reasoning_steps(generated_text)
        if not reasoning_steps:
            return None

        # Analyze each step
        step_analyses = []
        for step in reasoning_steps:
            analysis = self.analyze_reasoning_step(step, metrics_list, input_query, len(reasoning_steps))
            step_analyses.append({"step_info": step, "analysis": analysis})

        # Compile overall assessment
        return {
            "reasoning_analysis": {
                "steps": step_analyses,
                "overall_metrics": {
                    "total_steps": len(reasoning_steps),
                    "average_raw_entropy": sum(m.raw_entropy for m in metrics_list) / len(metrics_list),
                    "average_semantic_entropy": sum(m.semantic_entropy for m in metrics_list) / len(metrics_list),
                    "reasoning_flow_score": self._calculate_flow_score(step_analyses),
                },
            }
        }

    def _calculate_flow_score(self, step_analyses: List[Dict]) -> float:
        """Calculate how well reasoning steps flow together"""
        try:
            # Update this to match the actual JSON structure we receive
            coherence_scores = [
                float(step["analysis"].get("training_insights", {}).get("step_quality", {}).get("coherence", "0.5"))
                for step in step_analyses
            ]
            return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0.0
        except (KeyError, ValueError, AttributeError) as e:
            print(f"Error calculating flow score: {e}")
            return 0.0


class VLMAnalyzer(EntropyAnalyzer):
    def __init__(
        self,
        vision_config: Optional[Any] = None,
        use_cls_token: bool = True,
        min_token_prob: float = 0.01,
        insight_model: Optional[Any] = None,
        insight_tokenizer: Optional[Any] = None,
        insight_api_key: Optional[str] = None,
        insight_prompt_template: Optional[str] = None,
        insight_response_model: Optional[BaseModel] = VLMAnalysisResponseModel,
    ):
        # First call parent's __init__ with only the arguments it expects
        super().__init__(
            min_token_prob=min_token_prob,
            insight_model=insight_model,
            insight_tokenizer=insight_tokenizer,
            insight_api_key=insight_api_key,
            insight_prompt_template=insight_prompt_template,
            insight_response_model=insight_response_model,
        )

        # Then handle VLMAnalyzer-specific initialization
        self.patch_size = getattr(vision_config, "patch_size", None)
        self.image_size = getattr(vision_config, "image_size", None)
        self.use_cls_token = use_cls_token

        # Template that uses {{ }} for literal curly braces in the JSON structure
        self.vlm_analysis_template = VLM_ANALYSIS_PROMPT_TEMPLATE

    def set_vision_config(self, vision_config):
        """Set or update vision configuration"""
        self.patch_size = vision_config.patch_size
        self.image_size = vision_config.image_size

    def visualize_attention(
        self, attention_data: AttentionData, image: Image.Image, save_path: Optional[str] = None
    ) -> None:
        """Visualize attention patterns matching the Colab implementation"""
        try:
            # Calculate image dimensions and adjust extent
            img_width = image.width
            img_height = image.height

            # Calculate offset to center the attention map (matching your Colab)
            width_offset = img_width * 0.07  # 7% offset as in your code
            height_offset = img_height * 0.07

            # Create figure
            plt.figure(figsize=(15, 10))

            # Display original image
            plt.subplot(2, 1, 1)
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis("off")

            # Display attention overlay
            plt.subplot(2, 1, 2)
            plt.imshow(image)  # Base image first

            if attention_data.cumulative_attention is not None:
                attention_map = attention_data.cumulative_attention

                # Ensure correct shape
                grid_size = int((self.image_size // self.patch_size))
                if attention_map.shape != (grid_size, grid_size):
                    attention_map = attention_map.reshape(grid_size, grid_size)

                # Normalize attention map
                attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

                # Create attention overlay with same parameters as your Colab
                heatmap = plt.imshow(
                    attention_map,
                    cmap="viridis",
                    alpha=0.7,
                    interpolation="nearest",
                    extent=[-width_offset, img_width - width_offset, img_height + height_offset, -height_offset],
                )

                plt.colorbar(heatmap, fraction=0.046, pad=0.04)

            plt.title("Cumulative Attention Map")
            plt.axis("off")

            # Adjust layout
            plt.tight_layout(pad=3.0)

            # Save visualization
            if save_path:
                plt.savefig(save_path, bbox_inches="tight", dpi=300)
                print(f"Visualization saved to: {save_path}")
            else:
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, "attention_viz_temp.png")
                plt.savefig(temp_path, bbox_inches="tight", dpi=300)
                print(f"Visualization saved to: {temp_path}")

            plt.close("all")

        except Exception as e:
            print(f"Error during visualization: {str(e)}")
            traceback.print_exc()

    def process_attention_maps(self, attention_tensors, tokens):
        """Process attention maps for visualization"""
        if self.patch_size is None or self.image_size is None:
            raise ValueError("Vision config not set. Call set_vision_config with model's vision_config first.")

        num_patches = (self.image_size // self.patch_size) ** 2
        grid_size = int(num_patches**0.5)
        cumulative_attention = None
        token_attentions = []

        for token_idx, token in enumerate(tokens):
            try:
                # Extract attention for current token
                if token_idx >= len(attention_tensors):
                    continue

                step_attention = attention_tensors[token_idx]
                step_attentions = []

                # Process each attention layer
                for layer_attn in step_attention:
                    if isinstance(layer_attn, torch.Tensor):
                        # Get attention weights for image tokens
                        image_attention = layer_attn[0, :, -1, : num_patches + 1]
                        step_attentions.append(image_attention.mean(dim=0))

                if not step_attentions:
                    continue

                # Average attention across layers
                avg_attention = torch.stack(step_attentions).mean(dim=0)

                # Remove CLS token if present
                if self.use_cls_token:
                    avg_attention = avg_attention[1:]

                # Ensure correct shape and reshape
                avg_attention = avg_attention[:num_patches]
                attention_grid = avg_attention.reshape(grid_size, grid_size)

                # Store token-specific attention
                token_attentions.append({"token": token, "attention_grid": attention_grid.cpu().numpy()})

                # Update cumulative attention
                if cumulative_attention is None:
                    cumulative_attention = attention_grid.cpu().numpy()
                else:
                    cumulative_attention += attention_grid.cpu().numpy()

            except Exception as e:
                print(f"Error processing token {token_idx}: {str(e)}")
                continue

        # Normalize cumulative attention
        if cumulative_attention is not None:
            min_val = np.min(cumulative_attention)
            max_val = np.max(cumulative_attention)
            if max_val > min_val:
                cumulative_attention = (cumulative_attention - min_val) / (max_val - min_val)

        return AttentionData(cumulative_attention=cumulative_attention, token_attentions=token_attentions)

    def generate_overall_insight(
        self,
        metrics_list: List[UncertaintyMetrics],
        input_query: Optional[str] = "",
        generated_text: Optional[str] = "",
        attention_data: Optional[AttentionData] = None,
    ) -> Optional[Dict]:
        """Generate comprehensive analysis including both uncertainty and attention patterns"""
        if not self.together_model:
            return None

        # Format token metrics
        detailed_metrics = []
        for idx, metrics in enumerate(metrics_list):
            top_predictions = [f"{t.token} ({t.probability:.3f})" for t in metrics.token_predictions[:3]]
            step_metrics = (
                f"Step {idx}:\n"
                f"- Raw Entropy: {metrics.raw_entropy:.4f}\n"
                f"- Semantic Entropy: {metrics.semantic_entropy:.4f}\n"
                f"- Top 3 Predictions: {' | '.join(top_predictions)}"
            )
            detailed_metrics.append(step_metrics)

        # Format attention patterns
        attention_patterns = []
        if attention_data and attention_data.token_attentions:
            for attn in attention_data.token_attentions:
                token = attn["token"]
                grid = attn["attention_grid"]
                max_attention = np.max(grid)
                attention_patterns.append(f"Token '{token}': Max attention {max_attention:.4f}")

        # Generate insight using template
        prompt = self.vlm_analysis_template.format(
            detailed_metrics="\n\n".join(detailed_metrics),
            attention_patterns="\n".join(attention_patterns),
            input_query=input_query,
            generated_text=generated_text,
        )

        return self.together_model.generate_insight(prompt, self.insight_response_model)


# analyze images with VLM
class EnhancedVLMAnalyzer(VLMAnalyzer):
    def __init__(
        self,
        vision_config: Optional[Any] = None,
        use_cls_token: bool = True,
        min_token_prob: float = 0.01,
        insight_model: Optional[Any] = None,
        insight_tokenizer: Optional[Any] = None,
        insight_api_key: Optional[str] = None,
        insight_prompt_template: Optional[str] = None,
        visual_insight_response_model: BaseModel = EnhancedVLMAnalysisResponseModel,
    ):
        super().__init__(
            vision_config=vision_config,
            use_cls_token=use_cls_token,
            min_token_prob=min_token_prob,
            insight_model=insight_model,
            insight_tokenizer=insight_tokenizer,
            insight_api_key=insight_api_key,
            insight_prompt_template=insight_prompt_template,
        )

        # Initialize TogetherModelWrapper if insight_model is provided
        if isinstance(insight_model, str) and insight_api_key:
            if insight_model.startswith("together:"):
                model_name = insight_model.replace("together:", "")
                self.together_model = TogetherModelWrapper(model_name=model_name, api_key=insight_api_key)
        else:
            self.together_model = None

        self.visual_analysis_template = ENHANCED_VLM_ANALYSIS_PROMPT_TEMPLATE
        self.visual_insight_response_model = visual_insight_response_model

    def _encode_image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        import base64
        import io

        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()

    def _create_attention_visualization(
        self,
        image: Image.Image,
        attention_data: AttentionData,
    ) -> Image.Image:
        """Create visualization of attention overlay and return as PIL Image"""
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            # Uses the parent class's visualize_attention method to create the overlay
            self.visualize_attention(attention_data, image, tmp_file.name)
            viz_image = Image.open(tmp_file.name)
            os.unlink(tmp_file.name)
            return viz_image

    def generate_visual_insight(
        self,
        metrics_list: List[UncertaintyMetrics],
        image: Image.Image,
        attention_data: AttentionData,
        input_query: str,
        generated_text: str,
    ) -> Dict[str, Any]:
        """Generate insight using the model with attention visualization"""
        if not self.together_model:
            raise ValueError("No insight model configured")

        try:
            # Format token metrics
            detailed_metrics = []
            for idx, metrics in enumerate(metrics_list):
                top_predictions = [f"{t.token} ({t.probability:.3f})" for t in metrics.token_predictions[:3]]
                step_metrics = (
                    f"Step {idx}:\n"
                    f"- Raw Entropy: {metrics.raw_entropy:.4f}\n"
                    f"- Semantic Entropy: {metrics.semantic_entropy:.4f}\n"
                    f"- Top 3 Predictions: {' | '.join(top_predictions)}"
                )
                detailed_metrics.append(step_metrics)

            # Format attention patterns
            attention_patterns = []
            if attention_data and attention_data.token_attentions:
                for attn in attention_data.token_attentions:
                    token = attn["token"]
                    grid = attn["attention_grid"]
                    max_attention = np.max(grid)
                    attention_patterns.append(f"Token '{token}': Max attention {max_attention:.4f}")

            # Create and encode attention visualization
            viz_image = self._create_attention_visualization(image, attention_data)
            attention_base64 = self._encode_image_to_base64(viz_image)

            # Prepare the prompt
            prompt = self.visual_analysis_template.format(
                input_query=input_query,
                generated_text=generated_text,
                detailed_metrics="\n".join(detailed_metrics),
                attention_patterns="\n".join(attention_patterns),
            )

            # Use the wrapper to generate insight with the attention visualization
            return self.together_model.generate_insight_with_image(
                prompt=prompt,
                image_data=[attention_base64],
                temperature=0.7,
                max_tokens=800,
                response_model=self.visual_insight_response_model,
            )

        except Exception as e:
            print(f"Error in visual insight generation: {str(e)}")
            traceback.print_exc()
            return None

    def generate_overall_insight(
        self,
        metrics_list: List[UncertaintyMetrics],
        input_query: Optional[str] = "",
        generated_text: Optional[str] = "",
        attention_data: Optional[AttentionData] = None,
        image: Optional[Image.Image] = None,
        use_visual_analysis: bool = True,
    ) -> Optional[Dict]:
        """Generate comprehensive analysis including visual analysis by default"""
        # Get base analysis
        base_insight = super().generate_overall_insight(
            metrics_list=metrics_list,
            input_query=input_query,
            generated_text=generated_text,
            attention_data=attention_data,
        )

        # Convert string insight to dict if necessary
        final_insight = {}
        if isinstance(base_insight, str):
            try:
                # Try to parse as JSON first
                final_insight = json.loads(base_insight)
            except json.JSONDecodeError:
                # If not valid JSON, store as raw text
                final_insight = {"base_analysis": base_insight}
        elif isinstance(base_insight, dict):
            final_insight = base_insight
        elif base_insight is None:
            final_insight = {}

        # Add enhanced visual analysis if possible
        if use_visual_analysis and self.together_model and image and attention_data:
            visual_analysis = self.generate_visual_insight(
                metrics_list=metrics_list,
                image=image,
                attention_data=attention_data,
                input_query=input_query,
                generated_text=generated_text,
            )

            if visual_analysis:
                final_insight["enhanced_visual_analysis"] = visual_analysis

        return final_insight
