#!/usr/bin/env python3
"""
Training Data for GeometricLCM

Pre-defined training examples for gradient-free learning.
These examples teach the model about character roles, qualities, actions, and relations.

Usage:
    from truthspace_lcm.training_data import TRAINING_EXAMPLES, train_model
    
    qa = ConceptQA()
    qa.load_corpus('concept_corpus.json')
    train_model(qa)

Author: Lesley Gushurst
License: GPLv3
"""

from typing import List, Tuple

# Training examples: (entity, source, target_answer)
TRAINING_EXAMPLES: List[Tuple[str, str, str]] = [
    # ==========================================================================
    # SHERLOCK HOLMES CHARACTERS
    # ==========================================================================
    ('holmes', 'Sherlock Holmes', 
     'Holmes is a brilliant detective from Sherlock Holmes who investigates with Watson.'),
    ('watson', 'Sherlock Holmes', 
     'Watson is a loyal doctor from Sherlock Holmes who assists Holmes.'),
    ('moriarty', 'Sherlock Holmes',
     'Moriarty is a cunning villain from Sherlock Holmes who challenges Holmes.'),
    ('lestrade', 'Sherlock Holmes',
     'Lestrade is an inspector from Sherlock Holmes who helps Holmes.'),
    ('mycroft', 'Sherlock Holmes',
     'Mycroft is a clever gentleman from Sherlock Holmes who assists Holmes.'),
    ('irene', 'Sherlock Holmes',
     'Irene is a clever lady from Sherlock Holmes who challenges Holmes.'),
    
    # ==========================================================================
    # PRIDE AND PREJUDICE CHARACTERS
    # ==========================================================================
    ('darcy', 'Pride and Prejudice', 
     'Darcy is a proud gentleman from Pride and Prejudice who loves Elizabeth.'),
    ('elizabeth', 'Pride and Prejudice', 
     'Elizabeth is a witty lady from Pride and Prejudice who challenges Darcy.'),
    ('jane', 'Pride and Prejudice',
     'Jane is a kind lady from Pride and Prejudice who loves Bingley.'),
    ('bingley', 'Pride and Prejudice',
     'Bingley is a charming gentleman from Pride and Prejudice who admires Jane.'),
    ('wickham', 'Pride and Prejudice',
     'Wickham is a cunning villain from Pride and Prejudice who deceives Elizabeth.'),
    ('charlotte', 'Pride and Prejudice',
     'Charlotte is a practical friend from Pride and Prejudice who helps Elizabeth.'),
    ('collins', 'Pride and Prejudice',
     'Collins is a pompous gentleman from Pride and Prejudice.'),
    ('lydia', 'Pride and Prejudice',
     'Lydia is a foolish lady from Pride and Prejudice.'),
]


def train_model(qa, verbose: bool = True) -> dict:
    """
    Train the QA model with pre-defined examples.
    
    Args:
        qa: ConceptQA instance with corpus loaded
        verbose: Print training progress
    
    Returns:
        Training result dict with epochs and final_overlap
    """
    from .core.learnable_structure import train_from_examples
    
    # Initialize learnable structure if not already done
    if qa.projector.answer_generator.learnable is None:
        qa.projector.answer_generator.init_learnable()
    
    if verbose:
        print("Training GeometricLCM with quality examples...")
    
    # Train
    result = train_from_examples(
        qa.projector.answer_generator.learnable,
        TRAINING_EXAMPLES,
        max_epochs=5,
        target_overlap=0.95
    )
    
    if verbose:
        print(f"  Epochs: {result['epochs']}")
        print(f"  Final overlap: {result['final_overlap']:.1%}")
        stats = qa.projector.answer_generator.learnable.get_stats()
        print(f"  Learned: {stats['entities']} entities, "
              f"{stats['roles_learned']} roles, "
              f"{stats['qualities_learned']} qualities, "
              f"{stats['actions_learned']} actions")
    
    return result


def demo_training():
    """Demonstrate training and show before/after comparison."""
    from . import ConceptQA
    
    print("=" * 70)
    print("GeometricLCM Training Demo")
    print("=" * 70)
    print()
    
    qa = ConceptQA()
    qa.load_corpus('truthspace_lcm/concept_corpus.json')
    
    # Show before
    print("## Before Training")
    print()
    for entity in ['holmes', 'watson', 'darcy', 'elizabeth']:
        answer = qa.ask(f'Who is {entity}?')
        print(f"  {entity.title()}: {answer}")
    print()
    
    # Train
    print("## Training...")
    result = train_model(qa)
    print()
    
    # Show after
    print("## After Training")
    print()
    for entity in ['holmes', 'watson', 'darcy', 'elizabeth', 'wickham', 'moriarty']:
        answer = qa.ask(f'Who is {entity}?')
        print(f"  {entity.title()}: {answer}")
    print()
    
    print("=" * 70)
    print("Training complete!")
    print("=" * 70)


if __name__ == '__main__':
    demo_training()
