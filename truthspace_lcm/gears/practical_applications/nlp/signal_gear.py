"""
Signal Gear

Applies signal corpus patterns to the state.

Author: Lesley Gushurst
License: GPLv3
"""

import os
import json
import re
from typing import Dict, Any
from collections import Counter, defaultdict

from truthspace_lcm.gears.core import Gear, GearState, Quaternion


class SignalGear(Gear):
    """
    Applies signal corpus patterns to the state.
    
    Instead of replacing output with signal frames, this gear:
    1. Learns style patterns from signal corpus
    2. Applies those patterns to transform the state
    3. Passes the transformed state to the next gear
    """
    
    def __init__(self, signal_corpus_path: str = None, ratio: float = 1.0):
        super().__init__("SignalGear", ratio)
        
        self.patterns: Dict[str, Any] = {
            'prefixes': Counter(),
            'connectors': Counter(),
            'target_connectors': Counter(),
            'suffixes': Counter(),
            'role_phrases': defaultdict(Counter),
        }
        
        self.style_quaternions = {
            'formal': Quaternion(0.9, 0.1, 0.0, 0.0),
            'casual': Quaternion(0.7, 0.3, 0.1, 0.0),
            'technical': Quaternion(0.8, 0.1, 0.5, 0.0),
            'narrative': Quaternion(0.6, 0.2, 0.1, 0.5),
        }
        
        if signal_corpus_path and os.path.exists(signal_corpus_path):
            self._learn_patterns(signal_corpus_path)
    
    def _learn_patterns(self, corpus_path: str):
        """Learn style patterns from signal corpus."""
        with open(corpus_path, 'r') as f:
            data = json.load(f)
        
        for frame in data.get('frames', []):
            text = frame.get('text', '')
            text_lower = text.lower()
            
            # Learn prefixes
            if text.startswith('It seems'):
                self.patterns['prefixes']['it_seems'] += 1
            elif text.startswith('It appears'):
                self.patterns['prefixes']['it_appears'] += 1
            else:
                self.patterns['prefixes']['direct'] += 1
            
            # Learn connectors
            if 'that involves' in text_lower:
                self.patterns['connectors']['that_involves'] += 1
            elif 'who' in text_lower and 'is a' in text_lower:
                self.patterns['connectors']['who'] += 1
            elif 'that' in text_lower:
                self.patterns['connectors']['that'] += 1
            
            # Learn target connectors
            if 'particularly' in text_lower:
                self.patterns['target_connectors']['particularly'] += 1
            elif 'relating to' in text_lower:
                self.patterns['target_connectors']['relating_to'] += 1
            elif 'focusing on' in text_lower:
                self.patterns['target_connectors']['focusing_on'] += 1
    
    def forward(self, state: GearState) -> GearState:
        # Determine style based on role
        if state.role in ['detective', 'doctor', 'character']:
            state.signal_style = 'narrative'
        elif state.role in ['concept', 'field', 'science']:
            state.signal_style = 'technical'
        else:
            state.signal_style = 'formal'
        
        # Apply style quaternion
        if state.signal_style in self.style_quaternions:
            style_q = self.style_quaternions[state.signal_style]
            state.accumulated_q = state.accumulated_q * style_q
        
        # Preserve tense-specific connectors
        if state.connector in ["that will", "that has"]:
            pass  # Keep tense connector
        elif state.use_gerunds:
            state.connector = "that involves"
        elif self.patterns['connectors']:
            most_common = self.patterns['connectors'].most_common(1)
            if most_common:
                conn = most_common[0][0]
                if conn == 'who' and state.role in ['detective', 'doctor', 'character']:
                    state.connector = "who"
                elif conn == 'that_involves':
                    state.connector = "that involves"
                else:
                    state.connector = "that"
        
        # Determine target connector
        if self.patterns['target_connectors']:
            most_common = self.patterns['target_connectors'].most_common(1)
            if most_common:
                tc = most_common[0][0]
                if tc == 'particularly':
                    state.target_connector = "particularly"
                elif tc == 'focusing_on':
                    state.target_connector = "focusing on"
                else:
                    state.target_connector = "relating to"
        
        return state
