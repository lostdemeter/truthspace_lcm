"""
Serialization: Save and load discovered pipelines as JSON.

Supports saving a CascadeNavigator (the executable artifact) to JSON
and loading it back for execution on new inputs. This means you can:

    1. Discover a pipeline once from training pairs
    2. Save it to a file
    3. Load and execute it anywhere — no re-discovery needed

Usage:
    from phi_geometric import PhaseDiscovery
    from phi_geometric.core.serialization import save_pipeline, load_pipeline

    # Discover
    pd = PhaseDiscovery()
    pd.add_pair(['c', 'a', 't'], ['k', 'æ', 't'])
    result = pd.discover()
    nav = result.to_navigator()

    # Save
    save_pipeline(nav, 'my_pipeline.json')

    # Load (elsewhere, later)
    nav2 = load_pipeline('my_pipeline.json')
    trace = nav2.execute(['c', 'a', 'p'])

Author: TruthSpace LCM Project
Date: February 2026
"""

import json
from typing import Any, Dict, List, Optional, Hashable

from .discovery import TransformRule
from .cascade_navigator import (
    CascadeNavigator, Phase,
    default_context_extractor, geometric_context_extractor,
)

# JSON key for None values (since JSON null can't be a dict key)
_NULL_KEY = "__null__"


# =========================================================================
# JSON-safe conversion helpers
# =========================================================================

def _key_to_json(k: Hashable) -> str:
    """Convert a hashable key to a JSON-safe string key."""
    if k is None:
        return _NULL_KEY
    if isinstance(k, bool):
        return f"__bool__{k}"
    if isinstance(k, int):
        return f"__int__{k}"
    if isinstance(k, float):
        return f"__float__{k}"
    if isinstance(k, tuple):
        return f"__tuple__{json.dumps(list(k))}"
    return str(k)


def _key_from_json(s: str) -> Hashable:
    """Convert a JSON string key back to a hashable value."""
    if s == _NULL_KEY:
        return None
    if s.startswith("__bool__"):
        return s == "__bool__True"
    if s.startswith("__int__"):
        return int(s[7:])
    if s.startswith("__float__"):
        return float(s[9:])
    if s.startswith("__tuple__"):
        return tuple(json.loads(s[9:]))
    return s


def _val_to_json(v: Any) -> Any:
    """Convert a value to JSON-safe form."""
    if v is None:
        return None
    if isinstance(v, (str, int, float, bool)):
        return v
    if isinstance(v, tuple):
        return {"__tuple__": list(v)}
    if isinstance(v, set):
        return {"__set__": list(v)}
    if isinstance(v, list):
        return v
    return str(v)


def _val_from_json(v: Any) -> Any:
    """Convert a JSON value back to its original type."""
    if v is None:
        return None
    if isinstance(v, dict):
        if "__tuple__" in v:
            return tuple(v["__tuple__"])
        if "__set__" in v:
            return set(v["__set__"])
    return v


def _dict_to_json(d: Dict) -> Dict[str, Any]:
    """Convert a dict with hashable keys to JSON-safe form."""
    if d is None:
        return None
    return {_key_to_json(k): _val_to_json(v) for k, v in d.items()}


def _dict_from_json(d: Dict[str, Any]) -> Dict:
    """Convert a JSON dict back to original key/value types."""
    if d is None:
        return None
    return {_key_from_json(k): _val_from_json(v) for k, v in d.items()}


# =========================================================================
# TransformRule serialization
# =========================================================================

def _rule_to_dict(rule: TransformRule) -> Dict:
    """Serialize a TransformRule to a dict."""
    d = {
        "input_value": _val_to_json(rule.input_value),
        "rule_type": rule.rule_type,
    }

    params = rule.params
    if rule.rule_type == "consistent":
        d["output"] = _val_to_json(params["output"])

    elif rule.rule_type == "selector":
        d["variable"] = params["variable"]
        d["selector_map"] = _dict_to_json(params["selector_map"])
        if "default_output" in params:
            d["default_output"] = _val_to_json(params["default_output"])

    elif rule.rule_type == "geared":
        d["coarse_var"] = params["coarse_var"]
        d["pure_map"] = _dict_to_json(params["pure_map"])
        if "default_output" in params:
            d["default_output"] = _val_to_json(params["default_output"])
        if "fine_gears" in params and params["fine_gears"]:
            fg = {}
            for k, v in params["fine_gears"].items():
                fine_var, fine_map, channels, gain, zone_default = v
                fg[_key_to_json(k)] = {
                    "fine_var": fine_var,
                    "fine_map": _dict_to_json(fine_map) if fine_map else {},
                    "gain": gain,
                    "zone_default": _val_to_json(zone_default),
                }
            d["fine_gears"] = fg

    return d


def _rule_from_dict(d: Dict) -> TransformRule:
    """Deserialize a TransformRule from a dict."""
    input_value = _val_from_json(d["input_value"])
    rule_type = d["rule_type"]

    if rule_type == "identity":
        return TransformRule(input_value, "identity")

    elif rule_type == "consistent":
        return TransformRule(input_value, "consistent",
                             output=_val_from_json(d["output"]))

    elif rule_type == "selector":
        params = {
            "variable": d["variable"],
            "selector_map": _dict_from_json(d["selector_map"]),
        }
        if "default_output" in d:
            params["default_output"] = _val_from_json(d["default_output"])
        return TransformRule(input_value, "selector", **params)

    elif rule_type == "geared":
        params = {
            "coarse_var": d["coarse_var"],
            "pure_map": _dict_from_json(d["pure_map"]),
        }
        if "default_output" in d:
            params["default_output"] = _val_from_json(d["default_output"])
        if "fine_gears" in d:
            fg = {}
            for k_str, v in d["fine_gears"].items():
                k = _key_from_json(k_str)
                fine_map = _dict_from_json(v["fine_map"]) if v["fine_map"] else {}
                fg[k] = (
                    v["fine_var"],
                    fine_map,
                    {},  # channels not needed for execution
                    v["gain"],
                    _val_from_json(v["zone_default"]),
                )
            params["fine_gears"] = fg
        return TransformRule(input_value, "geared", **params)

    return TransformRule(input_value, rule_type)


# =========================================================================
# Phase serialization
# =========================================================================

def _phase_to_dict(phase: Phase) -> Dict:
    """Serialize a Phase to a dict."""
    # Detect which context extractor is used
    ctx_type = "default"
    if phase.context_extractor is geometric_context_extractor:
        ctx_type = "geometric"

    return {
        "name": phase.name,
        "rules": [_rule_to_dict(r) for r in phase.rules],
        "freeze_outputs": phase.freeze_outputs,
        "use_original_context": phase.use_original_context,
        "context_extractor": ctx_type,
    }


def _phase_from_dict(d: Dict) -> Phase:
    """Deserialize a Phase from a dict."""
    ctx_type = d.get("context_extractor", "default")
    extractor = None
    if ctx_type == "geometric":
        extractor = geometric_context_extractor

    phase = Phase(
        name=d["name"],
        freeze_outputs=d.get("freeze_outputs", False),
        use_original_context=d.get("use_original_context", True),
        context_extractor=extractor,
    )
    for rd in d["rules"]:
        phase.add_rule(_rule_from_dict(rd))
    return phase


# =========================================================================
# CascadeNavigator serialization
# =========================================================================

def navigator_to_dict(nav: CascadeNavigator) -> Dict:
    """Serialize a CascadeNavigator to a JSON-safe dict.

    The dict captures the full pipeline: collapse patterns, expand
    patterns, and all phases with their rules. It can be saved to
    JSON and loaded back to recreate an identical navigator.
    """
    return {
        "version": "2.0.0",
        "type": "CascadeNavigator",
        "collapse_patterns": [
            {
                "input": list(inp),
                "output": list(out),
                "freeze": freeze,
            }
            for inp, out, freeze in nav.collapse_patterns
        ],
        "expand_patterns": [
            {
                "input": _val_to_json(inp_tok),
                "output": list(out_toks),
            }
            for inp_tok, out_toks in nav.expand_patterns
        ],
        "phases": [_phase_to_dict(p) for p in nav.phases],
    }


def navigator_from_dict(d: Dict) -> CascadeNavigator:
    """Deserialize a CascadeNavigator from a dict."""
    nav = CascadeNavigator()

    for cp in d.get("collapse_patterns", []):
        nav.add_collapse(
            tuple(cp["input"]),
            tuple(cp["output"]),
            freeze=cp.get("freeze", False),
        )

    for ep in d.get("expand_patterns", []):
        nav.add_expand(
            _val_from_json(ep["input"]),
            tuple(ep["output"]),
        )

    for pd in d.get("phases", []):
        nav.add_phase(_phase_from_dict(pd))

    return nav


# =========================================================================
# Public API: save / load
# =========================================================================

def save_pipeline(nav: CascadeNavigator, path: str, indent: int = 2):
    """Save a CascadeNavigator pipeline to a JSON file.

    Args:
        nav: The navigator to save
        path: Output file path (e.g., 'my_pipeline.json')
        indent: JSON indentation (default 2, use None for compact)
    """
    d = navigator_to_dict(nav)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=indent, ensure_ascii=False)


def load_pipeline(path: str) -> CascadeNavigator:
    """Load a CascadeNavigator pipeline from a JSON file.

    Args:
        path: Input file path

    Returns:
        CascadeNavigator ready to execute
    """
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    return navigator_from_dict(d)
