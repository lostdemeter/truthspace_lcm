"""
Validation Gear

Validates data records against schema and rules.

Author: Lesley Gushurst
License: GPLv3
"""

from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass

from truthspace_lcm.gears.core import Gear, GearState, Quaternion


@dataclass
class ValidationRule:
    """A validation rule."""
    field: str
    rule_type: str  # 'required', 'type', 'range', 'pattern', 'custom'
    params: Dict[str, Any] = None
    message: str = ""


class ValidationGear(Gear):
    """
    Validates data records against schema and rules.
    
    Checks:
    - Required fields
    - Data types
    - Value ranges
    - Pattern matching
    - Custom validation functions
    
    Updates the quaternion quality score based on validation results.
    """
    
    def __init__(self, ratio: float = 1.0):
        super().__init__("ValidationGear", ratio)
        self.rules: List[ValidationRule] = []
        self.type_validators: Dict[str, Callable] = {
            'str': lambda x: isinstance(x, str),
            'int': lambda x: isinstance(x, int),
            'float': lambda x: isinstance(x, (int, float)),
            'bool': lambda x: isinstance(x, bool),
            'list': lambda x: isinstance(x, list),
            'dict': lambda x: isinstance(x, dict),
        }
    
    def add_required(self, field: str, message: str = None) -> 'ValidationGear':
        """Add a required field rule."""
        self.rules.append(ValidationRule(
            field=field,
            rule_type='required',
            message=message or f"Field '{field}' is required"
        ))
        return self
    
    def add_type(self, field: str, expected_type: str, message: str = None) -> 'ValidationGear':
        """Add a type validation rule."""
        self.rules.append(ValidationRule(
            field=field,
            rule_type='type',
            params={'type': expected_type},
            message=message or f"Field '{field}' must be of type {expected_type}"
        ))
        return self
    
    def add_range(self, field: str, min_val: float = None, max_val: float = None, 
                  message: str = None) -> 'ValidationGear':
        """Add a range validation rule."""
        self.rules.append(ValidationRule(
            field=field,
            rule_type='range',
            params={'min': min_val, 'max': max_val},
            message=message or f"Field '{field}' must be in range [{min_val}, {max_val}]"
        ))
        return self
    
    def add_pattern(self, field: str, pattern: str, message: str = None) -> 'ValidationGear':
        """Add a regex pattern validation rule."""
        self.rules.append(ValidationRule(
            field=field,
            rule_type='pattern',
            params={'pattern': pattern},
            message=message or f"Field '{field}' must match pattern {pattern}"
        ))
        return self
    
    def add_custom(self, field: str, validator: Callable[[Any], bool], 
                   message: str = None) -> 'ValidationGear':
        """Add a custom validation function."""
        self.rules.append(ValidationRule(
            field=field,
            rule_type='custom',
            params={'validator': validator},
            message=message or f"Field '{field}' failed custom validation"
        ))
        return self
    
    def _validate_record(self, record: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate a single record, return list of errors."""
        import re
        errors = []
        
        for rule in self.rules:
            field = rule.field
            value = record.get(field)
            
            if rule.rule_type == 'required':
                if value is None or value == '':
                    errors.append({
                        'field': field,
                        'rule': 'required',
                        'message': rule.message,
                        'value': value
                    })
            
            elif rule.rule_type == 'type' and value is not None:
                expected = rule.params['type']
                validator = self.type_validators.get(expected)
                if validator and not validator(value):
                    errors.append({
                        'field': field,
                        'rule': 'type',
                        'message': rule.message,
                        'value': value,
                        'expected': expected
                    })
            
            elif rule.rule_type == 'range' and value is not None:
                min_val = rule.params.get('min')
                max_val = rule.params.get('max')
                try:
                    num_val = float(value)
                    if min_val is not None and num_val < min_val:
                        errors.append({
                            'field': field,
                            'rule': 'range',
                            'message': rule.message,
                            'value': value
                        })
                    if max_val is not None and num_val > max_val:
                        errors.append({
                            'field': field,
                            'rule': 'range',
                            'message': rule.message,
                            'value': value
                        })
                except (ValueError, TypeError):
                    errors.append({
                        'field': field,
                        'rule': 'range',
                        'message': f"Field '{field}' is not numeric",
                        'value': value
                    })
            
            elif rule.rule_type == 'pattern' and value is not None:
                pattern = rule.params['pattern']
                if not re.match(pattern, str(value)):
                    errors.append({
                        'field': field,
                        'rule': 'pattern',
                        'message': rule.message,
                        'value': value
                    })
            
            elif rule.rule_type == 'custom' and value is not None:
                validator = rule.params['validator']
                if not validator(value):
                    errors.append({
                        'field': field,
                        'rule': 'custom',
                        'message': rule.message,
                        'value': value
                    })
        
        return errors
    
    def forward(self, state: GearState) -> GearState:
        """Validate all records in the state."""
        # Get records from state (DataState has .records, regular GearState uses metadata)
        records = getattr(state, 'records', state.metadata.get('records', []))
        
        valid = []
        invalid = []
        all_errors = []
        
        for record in records:
            errors = self._validate_record(record)
            if errors:
                invalid.append(record)
                all_errors.extend(errors)
            else:
                valid.append(record)
        
        # Update state
        if hasattr(state, 'valid_records'):
            state.valid_records = valid
            state.invalid_records = invalid
            state.validation_errors = all_errors
            state.transformations_applied.append('ValidationGear')
        else:
            state.metadata['valid_records'] = valid
            state.metadata['invalid_records'] = invalid
            state.metadata['validation_errors'] = all_errors
        
        # Update quality quaternion
        if records:
            validity_ratio = len(valid) / len(records)
            state.accumulated_q = Quaternion(
                w=validity_ratio,  # Overall quality
                x=state.accumulated_q.x,
                y=validity_ratio,  # Accuracy
                z=state.accumulated_q.z
            )
        
        return state
