#!/usr/bin/env python3
"""
COMPREHENSIVE BERT MODEL EVALUATION
===================================

This script evaluates BERT models using standard NLP metrics including:
- Perplexity
- Masked Language Modeling Accuracy
- Domain-specific Term Recognition
- Semantic Similarity
- Classification Performance (when fine-tuned)
- Intrinsic and Extrinsic Evaluation Metrics
"""

import torch
import numpy as np
from transformers import BertForMaskedLM, BertTokenizerFast, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, roc_auc_score, matthews_corrcoef
import json
import math
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class BERTEvaluator:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.load_model()
        
    def load_model(self):
        """Load the BERT model and tokenizer"""
        print(f"📂 Loading model from: {self.model_path}")
        try:
            self.model = BertForMaskedLM.from_pretrained(self.model_path)
            self.tokenizer = BertTokenizerFast.from_pretrained(self.model_path)
            self.model.eval()
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            
    def calculate_perplexity(self, test_texts):
        """
        Calculate perplexity - measures how well the model predicts text
        Lower perplexity = better model
        """
        print("\\n📊 CALCULATING PERPLEXITY")
        print("=" * 40)
        
        if not self.model or not self.tokenizer:
            print("❌ Model not loaded")
            return None
            
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for text in test_texts:
                # Tokenize
                inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                
                # Calculate loss
                outputs = self.model(**inputs, labels=inputs['input_ids'])
                loss = outputs.loss
                
                total_loss += loss.item() * inputs['input_ids'].size(1)
                total_tokens += inputs['input_ids'].size(1)
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"Perplexity: {perplexity:.4f}")
        
        # Interpretation
        if perplexity < 10:
            interpretation = "EXCELLENT - Very low perplexity"
        elif perplexity < 50:
            interpretation = "GOOD - Reasonable perplexity"
        elif perplexity < 100:
            interpretation = "MODERATE - Acceptable perplexity"
        else:
            interpretation = "HIGH - Model may need improvement"
            
        print(f"Interpretation: {interpretation}")
        
        return {
            'perplexity': perplexity,
            'avg_loss': avg_loss,
            'interpretation': interpretation
        }
    
    def evaluate_masked_lm_accuracy(self, test_cases):
        """
        Evaluate Masked Language Modeling accuracy
        Measures how often the model predicts the correct masked word
        """
        print("\\n🎭 EVALUATING MASKED LANGUAGE MODELING ACCURACY")
        print("=" * 60)
        
        if not self.model or not self.tokenizer:
            print("❌ Model not loaded")
            return None
            
        correct_predictions = 0
        total_predictions = 0
        top_k_accuracies = {1: 0, 3: 0, 5: 0, 10: 0}
        
        results = []
        
        with torch.no_grad():
            for test_case in test_cases:
                masked_text = test_case['masked_text']
                correct_word = test_case['correct_word'].lower().strip()
                
                # Tokenize
                inputs = self.tokenizer(masked_text, return_tensors='pt')
                
                # Find mask position
                mask_positions = torch.where(inputs['input_ids'] == self.tokenizer.mask_token_id)[1]
                
                if len(mask_positions) == 0:
                    continue
                    
                # Get predictions
                outputs = self.model(**inputs)
                predictions = outputs.logits
                
                mask_pos = mask_positions[0]
                mask_logits = predictions[0, mask_pos, :]
                
                # Get top predictions
                top_tokens = torch.topk(mask_logits, 10, dim=0).indices.tolist()
                predicted_words = [self.tokenizer.decode([token_id]).strip().lower() for token_id in top_tokens]
                
                # Check accuracy at different k values
                for k in [1, 3, 5, 10]:
                    if correct_word in predicted_words[:k]:
                        top_k_accuracies[k] += 1
                
                # Store result
                is_correct = correct_word in predicted_words[:1]
                if is_correct:
                    correct_predictions += 1
                    
                total_predictions += 1
                
                results.append({
                    'masked_text': masked_text,
                    'correct_word': correct_word,
                    'predicted_words': predicted_words[:5],
                    'is_correct': is_correct
                })
        
        # Calculate accuracies
        accuracies = {}
        for k in [1, 3, 5, 10]:
            accuracies[f'top_{k}_accuracy'] = top_k_accuracies[k] / total_predictions if total_predictions > 0 else 0
        
        print(f"Total test cases: {total_predictions}")
        print(f"Top-1 Accuracy: {accuracies['top_1_accuracy']:.4f} ({accuracies['top_1_accuracy']*100:.1f}%)")
        print(f"Top-3 Accuracy: {accuracies['top_3_accuracy']:.4f} ({accuracies['top_3_accuracy']*100:.1f}%)")
        print(f"Top-5 Accuracy: {accuracies['top_5_accuracy']:.4f} ({accuracies['top_5_accuracy']*100:.1f}%)")
        print(f"Top-10 Accuracy: {accuracies['top_10_accuracy']:.4f} ({accuracies['top_10_accuracy']*100:.1f}%)")
        
        return {
            'accuracies': accuracies,
            'results': results,
            'total_cases': total_predictions
        }
    
    def evaluate_domain_term_recognition(self):
        """
        Evaluate how well the model recognizes domain-specific terms
        """
        print("\\n🌿 EVALUATING DOMAIN TERM RECOGNITION")
        print("=" * 50)
        
        # Western Ghats biodiversity terms
        domain_terms = {
            'geographic': ['western', 'ghats', 'sahyadri', 'nilgiri', 'anamalai'],
            'taxonomic': ['species', 'endemic', 'genus', 'family', 'order'],
            'ecological': ['biodiversity', 'ecosystem', 'habitat', 'conservation'],
            'conservation': ['endangered', 'threatened', 'vulnerable', 'extinct'],
            'organisms': ['amphibian', 'reptile', 'mammal', 'bird', 'plant']
        }
        
        recognition_scores = {}
        
        for category, terms in domain_terms.items():
            recognized = 0
            total = len(terms)
            
            for term in terms:
                token_id = self.tokenizer.convert_tokens_to_ids(term)
                if token_id != self.tokenizer.unk_token_id:
                    recognized += 1
                    print(f"✅ {term} (ID: {token_id})")
                else:
                    print(f"❌ {term} (UNK)")
            
            recognition_rate = recognized / total
            recognition_scores[category] = {
                'recognized': recognized,
                'total': total,
                'rate': recognition_rate
            }
            
            print(f"\\n{category.upper()}: {recognized}/{total} ({recognition_rate*100:.1f}%)")
        
        # Overall recognition
        total_recognized = sum(scores['recognized'] for scores in recognition_scores.values())
        total_terms = sum(scores['total'] for scores in recognition_scores.values())
        overall_rate = total_recognized / total_terms
        
        print(f"\\nOVERALL DOMAIN TERM RECOGNITION: {total_recognized}/{total_terms} ({overall_rate*100:.1f}%)")
        
        return recognition_scores
    
    def evaluate_semantic_coherence(self):
        """
        Evaluate semantic coherence using word embeddings
        """
        print("\\n🧠 EVALUATING SEMANTIC COHERENCE")
        print("=" * 50)
        
        if not self.model or not self.tokenizer:
            print("❌ Model not loaded")
            return None
        
        # Test semantic relationships
        semantic_tests = [
            {
                'word1': 'biodiversity',
                'word2': 'conservation',
                'expected': 'high_similarity',
                'description': 'Biodiversity and conservation should be similar'
            },
            {
                'word1': 'endemic',
                'word2': 'species',
                'expected': 'high_similarity',
                'description': 'Endemic and species should be related'
            },
            {
                'word1': 'western',
                'word2': 'ghats',
                'expected': 'high_similarity',
                'description': 'Western and Ghats should be related'
            },
            {
                'word1': 'forest',
                'word2': 'habitat',
                'expected': 'medium_similarity',
                'description': 'Forest and habitat should be somewhat related'
            }
        ]
        
        similarities = []
        
        try:
            for test in semantic_tests:
                word1, word2 = test['word1'], test['word2']
                
                # Get embeddings
                inputs1 = self.tokenizer(word1, return_tensors='pt')
                inputs2 = self.tokenizer(word2, return_tensors='pt')
                
                with torch.no_grad():
                    outputs1 = self.model.bert(**inputs1)
                    outputs2 = self.model.bert(**inputs2)
                    
                    # Use [CLS] token embeddings
                    emb1 = outputs1.last_hidden_state[0, 0, :].numpy()
                    emb2 = outputs2.last_hidden_state[0, 0, :].numpy()
                    
                    # Calculate cosine similarity
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                    similarities.append(similarity)
                    
                    print(f"'{word1}' ↔ '{word2}': {similarity:.4f}")
                    print(f"   {test['description']}")
        
        except Exception as e:
            print(f"❌ Error calculating semantic similarity: {e}")
            return None
        
        avg_similarity = np.mean(similarities) if similarities else 0
        print(f"\\nAverage semantic similarity: {avg_similarity:.4f}")
        
        return {
            'similarities': similarities,
            'average': avg_similarity,
            'tests': semantic_tests
        }
    
    def create_classification_dataset(self):
        """
        Create a simple classification dataset for evaluation
        """
        # Sample biodiversity vs non-biodiversity texts
        biodiversity_texts = [
            "Western Ghats contains many endemic species",
            "Biodiversity conservation is crucial for ecosystem health",
            "Endemic amphibians face threats from habitat loss",
            "The forest ecosystem supports diverse wildlife",
            "Conservation efforts protect endangered species",
            "Tropical rainforests harbor exceptional biodiversity",
            "Endemic flora requires special protection measures",
            "Habitat fragmentation threatens species survival"
        ]
        
        non_biodiversity_texts = [
            "The weather forecast predicts rain tomorrow",
            "Technology advances improve daily life",
            "Economic policies affect market trends",
            "Sports events attract large audiences",
            "Educational reforms enhance learning outcomes",
            "Transportation systems connect cities efficiently",
            "Cultural festivals celebrate local traditions",
            "Business strategies drive company growth"
        ]
        
        # Create labels (1 for biodiversity, 0 for non-biodiversity)
        texts = biodiversity_texts + non_biodiversity_texts
        labels = [1] * len(biodiversity_texts) + [0] * len(non_biodiversity_texts)
        
        return texts, labels
    
    def evaluate_classification_performance(self):
        """
        Evaluate classification performance using embeddings
        """
        print("\\n📊 EVALUATING CLASSIFICATION PERFORMANCE")
        print("=" * 50)
        
        try:
            # Get classification dataset
            texts, true_labels = self.create_classification_dataset()
            
            # Extract embeddings
            embeddings = []
            
            with torch.no_grad():
                for text in texts:
                    inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                    outputs = self.model.bert(**inputs)
                    # Use [CLS] token embedding
                    embedding = outputs.last_hidden_state[0, 0, :].numpy()
                    embeddings.append(embedding)
            
            embeddings = np.array(embeddings)
            
            # Simple classification using similarity to biodiversity centroid
            bio_embeddings = embeddings[:len(texts)//2]  # First half are biodiversity
            non_bio_embeddings = embeddings[len(texts)//2:]  # Second half are non-biodiversity
            
            bio_centroid = np.mean(bio_embeddings, axis=0)
            non_bio_centroid = np.mean(non_bio_embeddings, axis=0)
            
            # Predict based on similarity to centroids
            predicted_labels = []
            for embedding in embeddings:
                bio_sim = np.dot(embedding, bio_centroid) / (np.linalg.norm(embedding) * np.linalg.norm(bio_centroid))
                non_bio_sim = np.dot(embedding, non_bio_centroid) / (np.linalg.norm(embedding) * np.linalg.norm(non_bio_centroid))
                
                predicted_labels.append(1 if bio_sim > non_bio_sim else 0)
            
            # Calculate metrics
            accuracy = accuracy_score(true_labels, predicted_labels)
            precision = precision_score(true_labels, predicted_labels, average='binary')
            recall = recall_score(true_labels, predicted_labels, average='binary')
            f1 = f1_score(true_labels, predicted_labels, average='binary')
            
            # Calculate specificity (True Negative Rate)
            tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = recall  # Sensitivity = Recall = True Positive Rate
            
            print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
            print(f"Precision: {precision:.4f} ({precision*100:.1f}%)")
            print(f"Recall/Sensitivity: {recall:.4f} ({recall*100:.1f}%)")
            print(f"Specificity: {specificity:.4f} ({specificity*100:.1f}%)")
            print(f"F1-Score: {f1:.4f}")
            
            # Matthews Correlation Coefficient
            mcc = matthews_corrcoef(true_labels, predicted_labels)
            print(f"Matthews Correlation Coefficient: {mcc:.4f}")
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'f1_score': f1,
                'mcc': mcc,
                'confusion_matrix': confusion_matrix(true_labels, predicted_labels)
            }
            
        except Exception as e:
            print(f"❌ Error in classification evaluation: {e}")
            return None
    
    def comprehensive_evaluation(self):
        """
        Run comprehensive evaluation of the BERT model
        """
        print("🔍 COMPREHENSIVE BERT MODEL EVALUATION")
        print("=" * 70)
        print(f"Model: {self.model_path}")
        print("=" * 70)
        
        results = {}
        
        # 1. Perplexity Evaluation
        test_texts = [
            "Western Ghats is a biodiversity hotspot in India.",
            "Endemic species require conservation efforts.",
            "The ecosystem contains diverse flora and fauna.",
            "Habitat loss threatens endangered species.",
            "Biodiversity conservation is crucial for sustainability."
        ]
        
        perplexity_results = self.calculate_perplexity(test_texts)
        results['perplexity'] = perplexity_results
        
        # 2. Masked Language Modeling Accuracy
        mlm_test_cases = [
            {'masked_text': 'Western Ghats contains many [MASK] species', 'correct_word': 'endemic'},
            {'masked_text': 'Biodiversity [MASK] is important', 'correct_word': 'conservation'},
            {'masked_text': 'Endemic [MASK] need protection', 'correct_word': 'species'},
            {'masked_text': 'The [MASK] supports diverse wildlife', 'correct_word': 'ecosystem'},
            {'masked_text': 'Conservation efforts protect [MASK] species', 'correct_word': 'endangered'},
            {'masked_text': 'Habitat [MASK] threatens biodiversity', 'correct_word': 'loss'},
            {'masked_text': 'Forest [MASK] contain endemic flora', 'correct_word': 'ecosystems'},
            {'masked_text': 'Threatened [MASK] require immediate action', 'correct_word': 'species'}
        ]
        
        mlm_results = self.evaluate_masked_lm_accuracy(mlm_test_cases)
        results['masked_lm'] = mlm_results
        
        # 3. Domain Term Recognition
        domain_results = self.evaluate_domain_term_recognition()
        results['domain_recognition'] = domain_results
        
        # 4. Semantic Coherence
        semantic_results = self.evaluate_semantic_coherence()
        results['semantic_coherence'] = semantic_results
        
        # 5. Classification Performance
        classification_results = self.evaluate_classification_performance()
        results['classification'] = classification_results
        
        # 6. Overall Assessment
        self.generate_overall_assessment(results)
        
        return results
    
    def generate_overall_assessment(self, results):
        """
        Generate overall assessment of model performance
        """
        print("\\n🎯 OVERALL MODEL ASSESSMENT")
        print("=" * 50)
        
        scores = []
        
        # Perplexity score (lower is better, normalize to 0-1)
        if results.get('perplexity'):
            perp = results['perplexity']['perplexity']
            perp_score = max(0, 1 - (perp - 1) / 99)  # Normalize assuming 1-100 range
            scores.append(('Perplexity', perp_score))
            print(f"Perplexity Score: {perp_score:.3f} (Perplexity: {perp:.2f})")
        
        # MLM Accuracy
        if results.get('masked_lm'):
            mlm_score = results['masked_lm']['accuracies']['top_1_accuracy']
            scores.append(('MLM Accuracy', mlm_score))
            print(f"MLM Accuracy Score: {mlm_score:.3f}")
        
        # Domain Recognition
        if results.get('domain_recognition'):
            domain_scores = [cat['rate'] for cat in results['domain_recognition'].values()]
            domain_score = np.mean(domain_scores)
            scores.append(('Domain Recognition', domain_score))
            print(f"Domain Recognition Score: {domain_score:.3f}")
        
        # Classification Performance
        if results.get('classification'):
            class_score = results['classification']['f1_score']
            scores.append(('Classification F1', class_score))
            print(f"Classification F1 Score: {class_score:.3f}")
        
        # Overall Score
        if scores:
            overall_score = np.mean([score for _, score in scores])
            print(f"\\n🏆 OVERALL MODEL SCORE: {overall_score:.3f} ({overall_score*100:.1f}%)")
            
            if overall_score >= 0.8:
                assessment = "EXCELLENT - Research-grade model"
            elif overall_score >= 0.6:
                assessment = "GOOD - Suitable for applications"
            elif overall_score >= 0.4:
                assessment = "MODERATE - Needs improvement"
            else:
                assessment = "POOR - Requires significant work"
            
            print(f"Assessment: {assessment}")
        
        return scores

def main():
    """Main evaluation function"""
    print("🧪 BERT MODEL EVALUATION FRAMEWORK")
    print("=" * 60)
    
    # Evaluate both models if available
    models_to_evaluate = [
        "western_ghats_bert",
        "western_ghats_bert_improved"
    ]
    
    for model_path in models_to_evaluate:
        if os.path.exists(model_path):
            print(f"\\n{'='*70}")
            print(f"EVALUATING: {model_path}")
            print(f"{'='*70}")
            
            evaluator = BERTEvaluator(model_path)
            results = evaluator.comprehensive_evaluation()
            
            # Save results
            results_file = f"{model_path}_evaluation_results.json"
            try:
                # Convert numpy types to Python types for JSON serialization
                def convert_numpy(obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, dict):
                        return {key: convert_numpy(value) for key, value in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy(item) for item in obj]
                    return obj
                
                results_serializable = convert_numpy(results)
                
                with open(results_file, 'w') as f:
                    json.dump(results_serializable, f, indent=2)
                print(f"\\n💾 Results saved to: {results_file}")
            except Exception as e:
                print(f"⚠️ Could not save results: {e}")
        else:
            print(f"⚠️ Model not found: {model_path}")

if __name__ == "__main__":
    import os
    main()