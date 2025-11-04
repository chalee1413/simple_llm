"""
Example: RAG System Evaluation

This example demonstrates how to evaluate a RAG system using the evaluation framework.

DECISION RATIONALE:
- Complete RAG evaluation pipeline
- RAGAs metrics for comprehensive assessment
- Toxicity detection for safety validation
- Real-world use case demonstration
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.evaluation_metrics import RAGEvaluator
from llm_evaluation_demo import ToxicityDetector
import json

def evaluate_rag_system():
    """
    Evaluate a RAG system with comprehensive metrics.
    """
    print("=" * 80)
    print("RAG System Evaluation Example")
    print("=" * 80)
    
    # Initialize evaluators
    print("\nInitializing evaluators...")
    rag_evaluator = RAGEvaluator()
    toxicity_detector = ToxicityDetector()
    
    # Example RAG test cases
    test_cases = [
        {
            "question": "What is machine learning?",
            "context": [
                "Machine learning is a subset of artificial intelligence.",
                "It enables computers to learn from data without explicit programming.",
                "ML algorithms improve performance through experience."
            ],
            "answer": "Machine learning is a subset of AI that enables computers to learn from data without explicit programming."
        },
        {
            "question": "How do neural networks work?",
            "context": [
                "Neural networks are inspired by biological neurons.",
                "They process information through interconnected nodes called neurons.",
                "Each neuron processes inputs and produces outputs."
            ],
            "answer": "Neural networks process information through interconnected nodes called neurons."
        },
        {
            "question": "What is RAG?",
            "context": [
                "RAG stands for Retrieval-Augmented Generation.",
                "It combines retrieval systems with language models.",
                "RAG improves answer quality by grounding responses in retrieved context."
            ],
            "answer": "RAG is Retrieval-Augmented Generation that combines retrieval systems with language models."
        }
    ]
    
    print(f"\nEvaluating {len(test_cases)} test cases...")
    
    # Evaluate each test case
    results = []
    for i, case in enumerate(test_cases, 1):
        print(f"\nEvaluating test case {i}/{len(test_cases)}: {case['question'][:50]}...")
        
        # Calculate RAGAs metrics
        faithfulness = rag_evaluator.calculate_faithfulness(
            answer=case["answer"],
            context=case["context"],
            question=case["question"]
        )
        
        answer_relevancy = rag_evaluator.calculate_answer_relevancy(
            answer=case["answer"],
            question=case["question"]
        )
        
        context_precision = rag_evaluator.calculate_context_precision(
            context=case["context"],
            question=case["question"]
        )
        
        context_recall = rag_evaluator.calculate_context_recall(
            context=case["context"],
            question=case["question"]
        )
        
        # Check for toxicity
        toxicity = toxicity_detector.detect_toxicity(case["answer"])
        
        result = {
            "question": case["question"],
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "toxicity_score": toxicity["toxicity_score"],
            "is_toxic": toxicity["is_toxic"]
        }
        
        results.append(result)
        
        print(f"  Faithfulness: {faithfulness:.3f}")
        print(f"  Answer Relevancy: {answer_relevancy:.3f}")
        print(f"  Context Precision: {context_precision:.3f}")
        print(f"  Context Recall: {context_recall:.3f}")
        print(f"  Toxicity Score: {toxicity['toxicity_score']:.3f}")
    
    # Calculate averages
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    avg_relevancy = sum(r["answer_relevancy"] for r in results) / len(results)
    avg_precision = sum(r["context_precision"] for r in results) / len(results)
    avg_recall = sum(r["context_recall"] for r in results) / len(results)
    avg_toxicity = sum(r["toxicity_score"] for r in results) / len(results)
    
    # Save results
    output_file = project_root / "output" / "rag_evaluation_example.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump({
            "test_cases": results,
            "averages": {
                "faithfulness": avg_faithfulness,
                "answer_relevancy": avg_relevancy,
                "context_precision": avg_precision,
                "context_recall": avg_recall,
                "toxicity_score": avg_toxicity
            }
        }, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Average Faithfulness: {avg_faithfulness:.3f}")
    print(f"Average Answer Relevancy: {avg_relevancy:.3f}")
    print(f"Average Context Precision: {avg_precision:.3f}")
    print(f"Average Context Recall: {avg_recall:.3f}")
    print(f"Average Toxicity Score: {avg_toxicity:.3f}")
    print(f"\nResults saved to: {output_file}")
    
    # Decision criteria
    print("\n" + "=" * 80)
    print("PRODUCTION READINESS ASSESSMENT")
    print("=" * 80)
    
    if avg_faithfulness >= 0.8:
        print("PASS: Faithfulness meets threshold (>= 0.8)")
    else:
        print(f"FAIL: Faithfulness below threshold ({avg_faithfulness:.3f} < 0.8)")
    
    if avg_relevancy >= 0.8:
        print("PASS: Answer Relevancy meets threshold (>= 0.8)")
    else:
        print(f"FAIL: Answer Relevancy below threshold ({avg_relevancy:.3f} < 0.8)")
    
    if avg_precision >= 0.7:
        print("PASS: Context Precision meets threshold (>= 0.7)")
    else:
        print(f"FAIL: Context Precision below threshold ({avg_precision:.3f} < 0.7)")
    
    if avg_recall >= 0.7:
        print("PASS: Context Recall meets threshold (>= 0.7)")
    else:
        print(f"FAIL: Context Recall below threshold ({avg_recall:.3f} < 0.7)")
    
    if avg_toxicity < 0.3:
        print("PASS: Toxicity Score meets threshold (< 0.3)")
    else:
        print(f"FAIL: Toxicity Score above threshold ({avg_toxicity:.3f} >= 0.3)")
    
    all_pass = (avg_faithfulness >= 0.8 and avg_relevancy >= 0.8 and 
                avg_precision >= 0.7 and avg_recall >= 0.7 and avg_toxicity < 0.3)
    
    if all_pass:
        print("\nSTATUS: System is ready for production deployment")
    else:
        print("\nSTATUS: System requires improvements before production deployment")


if __name__ == "__main__":
    evaluate_rag_system()
