# Design Consideration 011: Automated Knowledge Expansion

## Overview

We've achieved **100% accuracy on 50 queries** using pure geometric resolution with composable primitives. This document explores how to automate the expansion of the LCM's knowledge base—enabling it to learn new commands from sources like Linux man pages without manual code edits.

## What We've Learned

### The Composable Primitive Architecture

Our current system works by:

1. **Encoding**: Text → Primitive Signature (set of active primitives)
2. **Composition**: Primitive Signature → Command (via composition rules)
3. **Resolution**: Query matches the rule whose primitives best match

```
Query: "show disk space"
  ↓ encode
Primitives: {READ, STORAGE}
  ↓ match rules
Rule: (READ, STORAGE) → df
  ↓ resolve
Command: df
```

### Key Insights

1. **Primitives are the atoms of meaning**: ~25 primitives cover 50+ commands
2. **Composition is additive**: Commands = combinations of primitives
3. **Splitting resolves ambiguity**: When commands collide, split the primitive
4. **Exact match disambiguates**: +2 bonus for exact primitive match
5. **Language independent**: The geometry, not the words, determines meaning

### Current Primitive Structure

```
ACTIONS (dims 0-4):
  CREATE, DESTROY, READ, WRITE, COPY, RELOCATE, SEARCH,
  COMPRESS, SORT, FILTER, CONNECT, EXECUTE

DOMAINS (dims 5-7):
  FILE, DIRECTORY, SYSTEM, STORAGE, MEMORY, TIME, UPTIME, HOST,
  PROCESS, DATA, NETWORK, USER

MODIFIERS (dim 8):
  ALL, RECURSIVE, FORCE, VERBOSE

RELATIONS (dims 9-11):
  BEFORE, AFTER, DURING, CAUSE, EFFECT, IF, ELSE
```

## The Challenge: Automated Expansion

Currently, adding a new command requires:
1. Identifying which primitives it needs
2. Adding keywords to primitives (if missing)
3. Adding composition rules
4. Testing for regressions

We want to automate this so the LCM can:
1. Read a man page or documentation
2. Infer the primitive signature
3. Add the composition rule
4. Self-calibrate without breaking existing mappings

## Proposed Architecture

### Phase 1: Primitive Inference Engine

Given a new command's description, infer its primitive signature:

```python
class PrimitiveInferenceEngine:
    """Infer primitive signatures from command descriptions."""
    
    def infer_signature(self, command: str, description: str) -> Set[str]:
        """
        Given a command and its description, infer the primitive signature.
        
        Example:
            command: "lsof"
            description: "list open files"
            → {READ, FILE, PROCESS}  # files opened by processes
        """
        # Encode the description
        result = self.encoder.encode(description)
        primitives = set(p.name for p, _ in result.primitives)
        
        # Apply heuristics for common patterns
        primitives = self._apply_heuristics(command, description, primitives)
        
        return primitives
```

### Phase 2: Conflict Detection

Before adding a new rule, check for conflicts:

```python
class ConflictDetector:
    """Detect and resolve primitive signature conflicts."""
    
    def check_conflict(self, new_sig: Set[str], new_cmd: str) -> Optional[Conflict]:
        """
        Check if the new signature conflicts with existing rules.
        
        Returns:
            Conflict object if there's a collision, None otherwise
        """
        for rule in self.rules:
            if set(rule.primitives) == new_sig:
                return Conflict(
                    existing_cmd=rule.command,
                    new_cmd=new_cmd,
                    signature=new_sig,
                    resolution_needed=True
                )
        return None
    
    def suggest_resolution(self, conflict: Conflict) -> Resolution:
        """
        Suggest how to resolve a conflict.
        
        Options:
        1. Split a primitive (e.g., SYSTEM → SYSTEM + STORAGE)
        2. Add a distinguishing primitive
        3. Use context/modifiers
        """
        # Analyze the semantic difference between commands
        # Suggest which primitive to split or add
        pass
```

### Phase 3: Man Page Parser

Extract structured information from man pages:

```python
class ManPageParser:
    """Parse Linux man pages into structured command info."""
    
    def parse(self, man_content: str) -> CommandInfo:
        """
        Parse a man page into structured information.
        
        Returns:
            CommandInfo with:
            - name: command name
            - synopsis: usage pattern
            - description: what it does
            - examples: usage examples
            - related: related commands (SEE ALSO)
        """
        # Extract NAME section
        name = self._extract_section("NAME", man_content)
        
        # Extract DESCRIPTION
        description = self._extract_section("DESCRIPTION", man_content)
        
        # Extract EXAMPLES
        examples = self._extract_section("EXAMPLES", man_content)
        
        return CommandInfo(name, description, examples)
```

### Phase 4: Self-Calibration Loop

The core learning loop:

```python
class SelfCalibrator:
    """Self-calibrating knowledge expansion system."""
    
    def learn_command(self, command: str, man_page: str) -> LearningResult:
        """
        Learn a new command from its man page.
        
        Steps:
        1. Parse the man page
        2. Infer primitive signature
        3. Check for conflicts
        4. Resolve conflicts (split primitives if needed)
        5. Add composition rule
        6. Validate with test queries
        """
        # Step 1: Parse
        info = self.parser.parse(man_page)
        
        # Step 2: Infer signature
        signature = self.inference.infer_signature(command, info.description)
        
        # Step 3: Check conflicts
        conflict = self.detector.check_conflict(signature, command)
        
        if conflict:
            # Step 4: Resolve
            resolution = self.detector.suggest_resolution(conflict)
            self._apply_resolution(resolution)
            # Re-infer with new primitives
            signature = self.inference.infer_signature(command, info.description)
        
        # Step 5: Add rule
        self.resolver.add_rule(tuple(signature), command, info.description)
        
        # Step 6: Validate
        test_queries = self._generate_test_queries(info)
        results = self._run_tests(test_queries, command)
        
        return LearningResult(success=results.all_passed, signature=signature)
```

## Key Design Decisions

### 1. Primitive Vocabulary is Fixed (Initially)

The ~25 primitives we have are sufficient for most commands. New commands should map to existing primitives. Only when there's an unresolvable conflict do we split a primitive.

**Rationale**: Primitives are the "axes" of our semantic space. Adding new axes changes the geometry for everything. Splitting is safer than adding.

### 2. Keywords are Expandable

While primitives are fixed, their keywords can grow:

```python
# Current
STORAGE = {"disk", "space", "storage", "size", "filesystem"}

# After learning 'quota' command
STORAGE = {"disk", "space", "storage", "size", "filesystem", "quota"}
```

This is safe because keywords just help encoding—they don't change the geometric structure.

### 3. Composition Rules are Additive

New rules can be added without affecting existing ones (as long as signatures don't conflict):

```python
# Existing
(READ, STORAGE) → df

# New (no conflict - different signature)
(READ, STORAGE, USER) → quota
```

### 4. Conflict Resolution Strategy

When two commands have the same signature:

1. **First, try adding a distinguishing primitive**: 
   - `df` = READ + STORAGE
   - `quota` = READ + STORAGE + USER (quotas are per-user)

2. **If that fails, split a primitive**:
   - STORAGE → DISK_SPACE + QUOTA_SPACE

3. **If splitting would cause cascading changes, flag for human review**

### 5. Validation is Mandatory

Every new rule must pass validation:

```python
def validate_rule(self, command: str, signature: Set[str]) -> bool:
    """
    Validate a new rule doesn't break existing functionality.
    
    1. Generate test queries for the new command
    2. Run all existing test queries
    3. Check for regressions
    """
    # New command works?
    new_tests = self._generate_tests(command)
    new_pass = all(self._test(q, command) for q in new_tests)
    
    # Existing commands still work?
    regression_pass = all(self._test(q, exp) for q, exp in self.test_suite)
    
    return new_pass and regression_pass
```

## Implementation Roadmap

### Phase 1: Manual Expansion with Tooling (Current)

- [x] Composable primitive resolution
- [x] 50-query test suite
- [ ] CLI tool to add new commands with validation

### Phase 2: Semi-Automated Expansion

- [ ] Man page parser
- [ ] Primitive inference engine
- [ ] Conflict detection
- [ ] Human-in-the-loop for conflict resolution

### Phase 3: Fully Automated Expansion

- [ ] Automatic conflict resolution
- [ ] Self-calibration loop
- [ ] Regression test suite
- [ ] Confidence scoring for new rules

### Phase 4: Cross-Domain Expansion

- [ ] Extend beyond bash commands
- [ ] Python libraries, APIs, etc.
- [ ] Domain-specific primitive vocabularies

## Example: Learning `lsof`

Let's walk through how the system would learn `lsof`:

```
Input: man lsof

1. PARSE
   Name: lsof - list open files
   Description: "lsof lists information about files opened by processes"
   
2. INFER SIGNATURE
   Encode "list open files opened by processes"
   → {READ, FILE, PROCESS}
   
3. CHECK CONFLICTS
   Existing rules with {READ, FILE, PROCESS}? None.
   ✓ No conflict
   
4. ADD RULE
   (READ, FILE, PROCESS) → lsof
   
5. VALIDATE
   Test: "list open files" → lsof ✓
   Test: "show files opened by process" → lsof ✓
   Test: "what files is process using" → lsof ✓
   Regression: all 50 existing tests pass ✓
   
Result: Successfully learned lsof
```

## Example: Learning `quota` (with conflict)

```
Input: man quota

1. PARSE
   Name: quota - display disk usage and limits
   Description: "quota displays users' disk usage and limits"
   
2. INFER SIGNATURE
   Encode "display disk usage and limits"
   → {READ, STORAGE}  # Same as df!
   
3. CHECK CONFLICTS
   Existing: (READ, STORAGE) → df
   ⚠ CONFLICT DETECTED
   
4. SUGGEST RESOLUTION
   Analyze difference:
   - df: system-wide disk space
   - quota: per-user disk limits
   
   Suggestion: Add USER primitive to quota
   New signature: {READ, STORAGE, USER}
   
5. ADD RULE
   (READ, STORAGE, USER) → quota
   
6. VALIDATE
   Test: "show my disk quota" → quota ✓
   Test: "user disk limits" → quota ✓
   Test: "show disk space" → df ✓ (no regression)
   
Result: Successfully learned quota with disambiguation
```

## Open Questions

1. **How do we handle commands with multiple modes?**
   - `tar` can compress or extract
   - Solution: Multiple rules with different signatures?

2. **How do we handle command options/flags?**
   - `ls -la` vs `ls`
   - Solution: Modifiers as primitives?

3. **How do we prioritize when multiple rules match?**
   - Current: exact match bonus + longer rule preference
   - Need: confidence scoring?

4. **How do we handle domain-specific jargon?**
   - "inode", "symlink", "pipe"
   - Solution: Domain-specific keyword expansion?

5. **How do we validate semantic correctness?**
   - The system might learn a wrong mapping
   - Solution: Human review for low-confidence rules?

## Conclusion

The composable primitive architecture is inherently expandable. The key insight is that **primitives are stable, but their combinations are infinite**. By:

1. Keeping primitives fixed (or splitting carefully)
2. Expanding keywords freely
3. Adding composition rules additively
4. Validating against regression tests

We can build a self-calibrating system that learns new commands from documentation without manual code edits. The geometry does the heavy lifting—we just need to help it find the right intersection points.

---

*"The meaning lives at the intersection. We just need to teach the system where to look."*
