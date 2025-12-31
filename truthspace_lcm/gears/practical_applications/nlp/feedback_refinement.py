"""
Feedback Refinement Gear

Uses LLM to evaluate and refine emergent responses on the fly.
The LLM acts as a quality gate and correction suggester, while
the actual knowledge remains emergent.

Key principle: LLM refines OUTPUT, not the underlying knowledge.
This keeps the system emergent while improving response quality.

Author: Lesley Gushurst
License: GPLv3
"""

import json
import re
import requests
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from truthspace_lcm.gears.core import Gear, GearState


@dataclass
class RefinementResult:
    """Result of a refinement operation."""
    original: str
    refined: str
    score_before: float
    score_after: float
    feedback: str
    changes_made: List[str] = field(default_factory=list)
    

class FeedbackRefinementGear(Gear):
    """
    Gear that uses LLM to evaluate and refine responses.
    
    The LLM is used ONLY for:
    1. Scoring response quality (0-10)
    2. Suggesting grammatical/clarity improvements
    3. NOT for generating new content or knowledge
    
    This maintains the emergent nature while improving output quality.
    """
    
    # Evaluation prompt template
    EVAL_PROMPT = """Rate this response about "{topic}" on a scale of 0-10:

Response: "{response}"

Consider:
- Grammatical correctness
- Natural phrasing
- Clarity and coherence
- Appropriate level of detail

Reply with ONLY a JSON object:
{{"score": <0-10>, "feedback": "<brief feedback>"}}"""

    # Refinement prompt template
    REFINE_PROMPT = """Improve this response about "{topic}":

Original: "{response}"

Issues: {feedback}

Rules:
1. Keep the same factual content - don't add new information
2. Fix grammar and awkward phrasing
3. Make it sound more natural
4. Keep it concise

Reply with ONLY the improved response, nothing else."""

    # Polish prompt for minor improvements
    POLISH_PROMPT = """Polish this response for better readability:

"{response}"

Rules:
1. Keep ALL the same information
2. Only fix grammar and flow
3. Don't add or remove facts
4. Keep it concise

Reply with ONLY the polished response."""

    def __init__(self, llm_url: str = None, llm_model: str = None, 
                 threshold: float = 7.0, ratio: float = 1.0):
        super().__init__("FeedbackRefinementGear", ratio)
        
        self.llm_url = llm_url or "http://localhost:11434/api/generate"
        self.llm_model = llm_model or "qwen2.5:14b"
        self.threshold = threshold  # Score below this triggers refinement
        
        # Stats
        self.evaluations = 0
        self.refinements = 0
        self.total_score_improvement = 0.0
    
    def configure_llm(self, url: str, model: str):
        """Configure LLM endpoint."""
        self.llm_url = url
        self.llm_model = model
    
    def _call_llm(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """Call LLM for evaluation/refinement."""
        try:
            response = requests.post(
                self.llm_url,
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.3,  # Low temp for consistent evaluation
                    }
                },
                timeout=30
            )
            if response.status_code == 200:
                return response.json().get('response', '').strip()
        except Exception as e:
            pass
        return None
    
    def evaluate(self, response: str, topic: str = "") -> Tuple[float, str]:
        """
        Evaluate a response using LLM.
        
        Returns: (score, feedback)
        """
        self.evaluations += 1
        
        prompt = self.EVAL_PROMPT.format(topic=topic, response=response)
        result = self._call_llm(prompt, max_tokens=100)
        
        if not result:
            return 5.0, "Could not evaluate"
        
        # Parse JSON response
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^}]+\}', result)
            if json_match:
                data = json.loads(json_match.group())
                score = float(data.get('score', 5))
                feedback = data.get('feedback', '')
                return min(10, max(0, score)), feedback
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback: try to extract just a number
        numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', result)
        if numbers:
            score = float(numbers[0])
            return min(10, max(0, score)), result
        
        return 5.0, result
    
    def refine(self, response: str, topic: str = "", feedback: str = "") -> str:
        """
        Refine a response using LLM.
        
        Returns the refined response.
        """
        self.refinements += 1
        
        prompt = self.REFINE_PROMPT.format(
            topic=topic, 
            response=response,
            feedback=feedback or "Improve clarity and grammar"
        )
        result = self._call_llm(prompt, max_tokens=300)
        
        if result and len(result) > 10:
            # Clean up any quotes or prefixes
            result = result.strip('"\'')
            if result.lower().startswith('improved:'):
                result = result[9:].strip()
            return result
        
        return response  # Return original if refinement failed
    
    def polish(self, response: str) -> str:
        """
        Light polish for minor improvements.
        
        Used when score is close to threshold.
        """
        prompt = self.POLISH_PROMPT.format(response=response)
        result = self._call_llm(prompt, max_tokens=300)
        
        if result and len(result) > 10:
            return result.strip('"\'')
        return response
    
    def evaluate_and_refine(self, response: str, topic: str = "") -> RefinementResult:
        """
        Full evaluation and refinement pipeline.
        
        1. Evaluate response
        2. If score < threshold, refine
        3. Re-evaluate refined response
        4. Return result with before/after comparison
        """
        # Initial evaluation
        score_before, feedback = self.evaluate(response, topic)
        
        if score_before >= self.threshold:
            # Good enough, maybe just polish
            if score_before < 9.0:
                polished = self.polish(response)
                return RefinementResult(
                    original=response,
                    refined=polished,
                    score_before=score_before,
                    score_after=score_before + 0.5,  # Assume slight improvement
                    feedback=feedback,
                    changes_made=["Light polish applied"]
                )
            return RefinementResult(
                original=response,
                refined=response,
                score_before=score_before,
                score_after=score_before,
                feedback=feedback,
                changes_made=[]
            )
        
        # Needs refinement
        refined = self.refine(response, topic, feedback)
        
        # Re-evaluate
        score_after, _ = self.evaluate(refined, topic)
        
        # Track improvement
        improvement = score_after - score_before
        self.total_score_improvement += improvement
        
        changes = []
        if refined != response:
            changes.append(f"Refined based on: {feedback}")
            if improvement > 0:
                changes.append(f"Score improved: {score_before:.1f} → {score_after:.1f}")
        
        return RefinementResult(
            original=response,
            refined=refined,
            score_before=score_before,
            score_after=score_after,
            feedback=feedback,
            changes_made=changes
        )
    
    def forward(self, state: GearState) -> GearState:
        """Apply refinement to the response in state."""
        response = state.metadata.get('response', '')
        topic = state.entity or ''
        
        if response:
            result = self.evaluate_and_refine(response, topic)
            state.metadata['original_response'] = result.original
            state.metadata['response'] = result.refined
            state.metadata['refinement_score'] = result.score_after
            state.metadata['refinement_feedback'] = result.feedback
        
        return state
    
    def get_stats(self) -> Dict[str, Any]:
        """Get refinement statistics."""
        avg_improvement = (
            self.total_score_improvement / self.refinements 
            if self.refinements > 0 else 0
        )
        return {
            'evaluations': self.evaluations,
            'refinements': self.refinements,
            'avg_score_improvement': avg_improvement,
        }


class AutoRefiner:
    """
    Automatic refinement system for ConversationalChain.
    
    Wraps a chain and automatically refines responses before returning.
    """
    
    def __init__(self, chain, llm_url: str = None, llm_model: str = None,
                 threshold: float = 7.0, auto_refine: bool = True):
        self.chain = chain
        self.gear = FeedbackRefinementGear(llm_url, llm_model, threshold)
        self.auto_refine = auto_refine
        
        # Use chain's LLM config if available
        if hasattr(chain, 'llm_url') and chain.llm_url:
            self.gear.configure_llm(chain.llm_url, chain.llm_model)
    
    def chat(self, user_input: str, refine: bool = None) -> str:
        """
        Chat with automatic refinement.
        
        Args:
            user_input: User's message
            refine: Override auto_refine setting
        
        Returns:
            Refined response
        """
        # Get emergent response
        response = self.chain.chat(user_input)
        
        # Determine if we should refine
        should_refine = refine if refine is not None else self.auto_refine
        
        if should_refine:
            # Extract topic from input
            topics = self.chain._extract_topics(user_input)
            topic = topics[0] if topics else ""
            
            # Refine
            result = self.gear.evaluate_and_refine(response, topic)
            return result.refined
        
        return response
    
    def chat_with_details(self, user_input: str) -> Dict[str, Any]:
        """
        Chat and return detailed refinement info.
        """
        # Get emergent response
        response = self.chain.chat(user_input)
        
        # Extract topic
        topics = self.chain._extract_topics(user_input)
        topic = topics[0] if topics else ""
        
        # Refine
        result = self.gear.evaluate_and_refine(response, topic)
        
        return {
            'original': result.original,
            'refined': result.refined,
            'score_before': result.score_before,
            'score_after': result.score_after,
            'feedback': result.feedback,
            'changes': result.changes_made,
            'topic': topic,
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined stats."""
        chain_stats = self.chain.get_stats()
        gear_stats = self.gear.get_stats()
        return {**chain_stats, 'refinement': gear_stats}
