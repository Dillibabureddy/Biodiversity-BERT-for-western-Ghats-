#!/usr/bin/env python3
"""
DETAILED WESTERN GHATS BERT MODEL ANALYSIS
==========================================

This script performs detailed analysis to understand why the model
isn't predicting domain-specific terms correctly.
"""

import torch
from transformers import BertForMaskedLM, BertTokenizerFast
import numpy as np

def analyze_vocabulary():
    """Analyze the tokenizer vocabulary for Western Ghats terms"""
    print("🔍 ANALYZING VOCABULARY")
    print("=" * 40)
    
    try:
        tokenizer = BertTokenizerFast.from_pretrained("western_ghats_bert")
        
        # Check for key Western Ghats terms
        key_terms = [
            'biodiversity', 'endemic', 'species', 'conservation',
            'western', 'ghats', 'sahyadri', 'nilgiri',
            'amphibian', 'reptile', 'forest', 'habitat',
            'threatened', 'endangered', 'ecosystem'
        ]
        
        print("Checking key terms in vocabulary:")
        found_terms = []
        missing_terms = []
        
        for term in key_terms:
            token_id = tokenizer.convert_tokens_to_ids(term)
            if token_id != tokenizer.unk_token_id:
                found_terms.append((term, token_id))
                print(f"✅ '{term}' -> ID: {token_id}")
            else:
                missing_terms.append(term)
                print(f"❌ '{term}' -> UNK (not in vocab)")
        
        print(f"\\nSummary:")
        print(f"  Found: {len(found_terms)}/{len(key_terms)} terms")
        print(f"  Missing: {len(missing_terms)} terms")
        
        if missing_terms:
            print(f"  Missing terms: {missing_terms}")
        
        return found_terms, missing_terms
        
    except Exception as e:
        print(f"❌ Error analyzing vocabulary: {e}")
        return [], []

def test_different_contexts():
    """Test the model with different contexts and mask positions"""
    print("\\n🎭 TESTING DIFFERENT CONTEXTS")
    print("=" * 50)
    
    try:
        model = BertForMaskedLM.from_pretrained("western_ghats_bert")
        tokenizer = BertTokenizerFast.from_pretrained("western_ghats_bert")
        model.eval()
        
        test_sentences = [
            "Western Ghats is a [MASK] hotspot",
            "Many [MASK] species live in Western Ghats",
            "The [MASK] of Western Ghats is remarkable",
            "[MASK] species are found in Sahyadri mountains",
            "Western Ghats contains [MASK] forests",
            "Conservation of [MASK] is important",
            "Endemic [MASK] need protection",
            "Biodiversity [MASK] is crucial"
        ]
        
        for sentence in test_sentences:
            print(f"\\n📝 Testing: '{sentence}'")
            
            # Tokenize
            inputs = tokenizer(sentence, return_tensors='pt')
            
            # Find mask position
            mask_positions = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
            
            if len(mask_positions) == 0:
                print("   ⚠️ No [MASK] token found")
                continue
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = outputs.logits
            
            # Get top 10 predictions for mask token
            mask_pos = mask_positions[0]
            mask_logits = predictions[0, mask_pos, :]
            top_10_tokens = torch.topk(mask_logits, 10, dim=0).indices.tolist()
            
            print("   Top 10 predictions:")
            for i, token_id in enumerate(top_10_tokens):
                token = tokenizer.decode([token_id]).strip()
                probability = torch.softmax(mask_logits, dim=0)[token_id].item()
                print(f"   {i+1:2d}. '{token}' (prob: {probability:.4f})")
        
    except Exception as e:
        print(f"❌ Error testing contexts: {e}")

def analyze_training_effectiveness():
    """Analyze if the model learned domain-specific patterns"""
    print("\\n🧠 ANALYZING TRAINING EFFECTIVENESS")
    print("=" * 50)
    
    try:
        model = BertForMaskedLM.from_pretrained("western_ghats_bert")
        tokenizer = BertTokenizerFast.from_pretrained("western_ghats_bert")
        model.eval()
        
        # Test domain-specific vs generic contexts
        domain_contexts = [
            "Western Ghats biodiversity",
            "Endemic species conservation", 
            "Sahyadri mountain ecosystem",
            "Nilgiri biosphere reserve"
        ]
        
        generic_contexts = [
            "The weather is",
            "People like to",
            "This is a",
            "We can see"
        ]
        
        print("🌿 Domain-specific contexts:")
        for context in domain_contexts:
            inputs = tokenizer(context, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs)
                # Get average attention weights as a proxy for "understanding"
                attention_mean = outputs.logits.mean().item()
                print(f"   '{context}' -> avg logit: {attention_mean:.4f}")
        
        print("\\n📰 Generic contexts:")
        for context in generic_contexts:
            inputs = tokenizer(context, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs)
                attention_mean = outputs.logits.mean().item()
                print(f"   '{context}' -> avg logit: {attention_mean:.4f}")
        
    except Exception as e:
        print(f"❌ Error analyzing training: {e}")

def check_model_vs_base_bert():
    """Compare predictions with base BERT"""
    print("\\n⚖️ COMPARING WITH BASE BERT")
    print("=" * 40)
    
    try:
        # Load your model
        your_model = BertForMaskedLM.from_pretrained("western_ghats_bert")
        your_tokenizer = BertTokenizerFast.from_pretrained("western_ghats_bert")
        
        # Load base BERT
        base_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        base_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        
        test_sentence = "Western Ghats contains many [MASK] species"
        
        print(f"Test sentence: '{test_sentence}'")
        
        # Test your model
        print("\\n🌿 Your Western Ghats BERT:")
        inputs = your_tokenizer(test_sentence, return_tensors='pt')
        with torch.no_grad():
            outputs = your_model(**inputs)
        
        mask_pos = torch.where(inputs['input_ids'] == your_tokenizer.mask_token_id)[1][0]
        top_5 = torch.topk(outputs.logits[0, mask_pos, :], 5).indices.tolist()
        
        for i, token_id in enumerate(top_5):
            token = your_tokenizer.decode([token_id]).strip()
            print(f"   {i+1}. '{token}'")
        
        # Test base BERT
        print("\\n🤖 Base BERT:")
        inputs = base_tokenizer(test_sentence, return_tensors='pt')
        with torch.no_grad():
            outputs = base_model(**inputs)
        
        mask_pos = torch.where(inputs['input_ids'] == base_tokenizer.mask_token_id)[1][0]
        top_5 = torch.topk(outputs.logits[0, mask_pos, :], 5).indices.tolist()
        
        for i, token_id in enumerate(top_5):
            token = base_tokenizer.decode([token_id]).strip()
            print(f"   {i+1}. '{token}'")
        
    except Exception as e:
        print(f"❌ Error comparing models: {e}")

def diagnose_training_issues():
    """Diagnose potential training issues"""
    print("\\n🔧 DIAGNOSING POTENTIAL TRAINING ISSUES")
    print("=" * 50)
    
    issues_found = []
    
    # Check if model is actually different from base BERT
    try:
        your_model = BertForMaskedLM.from_pretrained("western_ghats_bert")
        base_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
        
        # Compare a few parameters
        your_params = list(your_model.parameters())[0].flatten()[:100]
        base_params = list(base_model.parameters())[0].flatten()[:100]
        
        param_diff = torch.mean(torch.abs(your_params - base_params)).item()
        
        print(f"Parameter difference from base BERT: {param_diff:.6f}")
        
        if param_diff < 0.001:
            issues_found.append("Model parameters very similar to base BERT - training may not have been effective")
        else:
            print("✅ Model parameters differ from base BERT - training occurred")
        
    except Exception as e:
        print(f"❌ Error checking parameters: {e}")
    
    # Check vocabulary usage
    found_terms, missing_terms = analyze_vocabulary()
    
    if len(missing_terms) > len(found_terms):
        issues_found.append("Many domain terms missing from vocabulary")
    
    # Summary
    print(f"\\n📋 DIAGNOSIS SUMMARY:")
    if issues_found:
        print("❌ Issues found:")
        for issue in issues_found:
            print(f"   - {issue}")
        
        print("\\n💡 Possible solutions:")
        print("   1. Model may need more training epochs")
        print("   2. Learning rate might be too low")
        print("   3. Domain vocabulary not well represented")
        print("   4. Need more domain-specific training data")
    else:
        print("✅ No obvious issues detected")
        print("   Model appears to have trained properly")

def main():
    """Main analysis function"""
    print("🔍 DETAILED WESTERN GHATS BERT MODEL ANALYSIS")
    print("=" * 70)
    print("Analyzing why masked predictions aren't domain-specific...")
    print("=" * 70)
    
    # Run comprehensive analysis
    analyze_vocabulary()
    test_different_contexts()
    analyze_training_effectiveness()
    check_model_vs_base_bert()
    diagnose_training_issues()
    
    print("\\n" + "=" * 70)
    print("🎯 ANALYSIS COMPLETE")
    print("=" * 70)
    print("\\nThe analysis above shows:")
    print("1. Which domain terms are in the vocabulary")
    print("2. How the model performs on different contexts")
    print("3. Whether training was effective")
    print("4. How it compares to base BERT")
    print("5. Potential issues and solutions")

if __name__ == "__main__":
    main()