"""
GeometricLCM Core Module.

Contains the orchestrator, handlers, self-knowledge, context, and generation config.
"""

from .orchestrator import Orchestrator, IntentClassifier
from .self_knowledge import SelfKnowledge, get_self_knowledge
from .query_resolver import QueryResolver, get_query_resolver
from .conversation_context import ConversationContext, get_conversation_context
from .response_templates import ResponseFormatter, get_formatter
from .generation_config import (
    GenerationConfig, 
    get_default_config, 
    set_default_config,
    create_config_from_request,
    get_preset,
    PRESETS,
)
from .response_length import (
    ResponseLengthController,
    IncrementalBuilder,
    create_length_controller,
    LengthStats,
)
from .natural_response import (
    NaturalResponseGenerator,
    get_natural_generator,
    generate_character_response,
    normalize_entity,
)
from .dynamic_profile import (
    DynamicProfileBuilder,
    DynamicProfile,
    get_profile_builder,
    generate_dynamic_response,
)

__all__ = [
    'Orchestrator',
    'IntentClassifier', 
    'SelfKnowledge',
    'get_self_knowledge',
    'QueryResolver',
    'get_query_resolver',
    'ConversationContext',
    'get_conversation_context',
    'ResponseFormatter',
    'get_formatter',
    'GenerationConfig',
    'get_default_config',
    'set_default_config',
    'create_config_from_request',
    'get_preset',
    'PRESETS',
    'ResponseLengthController',
    'IncrementalBuilder',
    'create_length_controller',
    'LengthStats',
    'NaturalResponseGenerator',
    'get_natural_generator',
    'generate_character_response',
    'normalize_entity',
    'DynamicProfileBuilder',
    'DynamicProfile',
    'get_profile_builder',
    'generate_dynamic_response',
]
