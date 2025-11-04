"""
Evaluation metrics for RAG systems following RAGAs framework patterns.

DECISION RATIONALE:
- RAGAs framework is the current SoTA for RAG evaluation (2024-2025)
- Implements faithfulness, answer relevancy, context precision, and context recall
- These metrics provide comprehensive assessment of RAG system quality
- Semantic similarity using sentence transformers for context-aware evaluation

References:
- RAGAs: Retrieval-Augmented Generation Assessment (2024)
  Es, S., S Parthasarathy, S., Talukdar, P., et al.
  https://arxiv.org/abs/2312.10997
- Evaluation metrics align with RAGAs framework methodology
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class RAGEvaluator:
    """
    RAG evaluation metrics following RAGAs framework patterns.
    
    DECISION RATIONALE:
    - Encapsulates evaluation logic in a reusable class
    - Uses sentence transformers for semantic similarity calculations
    - Implements RAGAs metrics: faithfulness, answer relevancy, context precision, context recall
    - Provides consistent interface for all RAG evaluation metrics
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG evaluator with sentence transformer model.
        
        Args:
            model_name: HuggingFace model name for sentence embeddings
                       Default: "all-MiniLM-L6-v2" (balanced speed/quality)
        
        DECISION RATIONALE:
        - all-MiniLM-L6-v2 provides good balance of speed and quality
        - Smaller model for faster inference during evaluation
        - Can be swapped for larger models if higher quality needed
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def calculate_faithfulness(
        self,
        answer: str,
        context: List[str],
        question: str
    ) -> float:
        """
        Calculate faithfulness score: whether answer is grounded in context.
        
        Faithfulness measures if the answer is supported by the provided context.
        Higher score indicates answer is more faithful to the context.
        
        Formula: Semantic similarity between answer and context chunks
        
        Args:
            answer: Generated answer from RAG system
            context: List of context chunks used for generation
            question: Original question
        
        Returns:
            float: Faithfulness score between 0 and 1
        
        DECISION RATIONALE:
        - Uses semantic similarity to measure answer-context alignment
        - Takes maximum similarity across all context chunks
        - Normalized to 0-1 range for interpretability
        - Follows RAGAs framework faithfulness calculation approach
        """
        if not context or not answer:
            return 0.0
        
        # Encode answer and context chunks
        answer_embedding = self.model.encode([answer], convert_to_numpy=True)
        context_embeddings = self.model.encode(context, convert_to_numpy=True)
        
        # Calculate cosine similarity between answer and each context chunk
        similarities = cosine_similarity(answer_embedding, context_embeddings)[0]
        
        # Faithfulness is the maximum similarity (answer is faithful if it matches any context)
        faithfulness_score = float(np.max(similarities))
        
        # Normalize to ensure 0-1 range (cosine similarity is already -1 to 1, but embeddings are typically 0-1)
        faithfulness_score = max(0.0, min(1.0, faithfulness_score))
        
        return faithfulness_score
    
    def calculate_answer_relevancy(
        self,
        answer: str,
        question: str
    ) -> float:
        """
        Calculate answer relevancy score: whether answer is relevant to question.
        
        Answer relevancy measures if the answer directly addresses the question.
        Higher score indicates answer is more relevant to the question.
        
        Formula: Semantic similarity between answer and question
        
        Args:
            answer: Generated answer from RAG system
            question: Original question
        
        Returns:
            float: Answer relevancy score between 0 and 1
        
        DECISION RATIONALE:
        - Uses semantic similarity to measure answer-question alignment
        - Higher similarity indicates more relevant answer
        - Normalized to 0-1 range for interpretability
        - Follows RAGAs framework answer relevancy calculation approach
        """
        if not answer or not question:
            return 0.0
        
        # Encode answer and question
        answer_embedding = self.model.encode([answer], convert_to_numpy=True)
        question_embedding = self.model.encode([question], convert_to_numpy=True)
        
        # Calculate cosine similarity
        relevancy_score = float(cosine_similarity(answer_embedding, question_embedding)[0][0])
        
        # Normalize to ensure 0-1 range
        relevancy_score = max(0.0, min(1.0, relevancy_score))
        
        return relevancy_score
    
    def calculate_context_precision(
        self,
        context: List[str],
        question: str,
        ground_truth: Optional[str] = None
    ) -> float:
        """
        Calculate context precision: proportion of relevant context chunks.
        
        Context precision measures how many retrieved context chunks are relevant.
        Higher score indicates more precise context retrieval.
        
        Formula: (Relevant chunks) / (Total chunks)
        
        Args:
            context: List of context chunks retrieved
            question: Original question
            ground_truth: Optional ground truth answer for relevance judgment
        
        Returns:
            float: Context precision score between 0 and 1
        
        DECISION RATIONALE:
        - Uses semantic similarity to determine chunk relevance
        - If ground truth provided, compares chunks to ground truth
        - Otherwise, compares chunks to question (assumes relevant chunks answer question)
        - Follows RAGAs framework context precision calculation approach
        """
        if not context:
            return 0.0
        
        # Use ground truth if available, otherwise use question
        reference_text = ground_truth if ground_truth else question
        
        # Encode reference and context chunks
        reference_embedding = self.model.encode([reference_text], convert_to_numpy=True)
        context_embeddings = self.model.encode(context, convert_to_numpy=True)
        
        # Calculate similarity between reference and each context chunk
        similarities = cosine_similarity(reference_embedding, context_embeddings)[0]
        
        # Threshold for relevance (chunks above threshold are considered relevant)
        relevance_threshold = 0.5
        relevant_chunks = np.sum(similarities >= relevance_threshold)
        
        # Context precision: proportion of relevant chunks
        context_precision = float(relevant_chunks / len(context))
        
        return context_precision
    
    def calculate_context_recall(
        self,
        context: List[str],
        question: str,
        ground_truth: Optional[str] = None
    ) -> float:
        """
        Calculate context recall: coverage of relevant information in context.
        
        Context recall measures how much of the relevant information is in context.
        Higher score indicates better recall of relevant information.
        
        Formula: Semantic similarity between ground truth and combined context
        
        Args:
            context: List of context chunks retrieved
            question: Original question
            ground_truth: Ground truth answer (required for recall calculation)
        
        Returns:
            float: Context recall score between 0 and 1
        
        DECISION RATIONALE:
        - Requires ground truth for meaningful recall calculation
        - Combines context chunks and compares to ground truth
        - Higher similarity indicates better recall
        - Follows RAGAs framework context recall calculation approach
        """
        if not context:
            return 0.0
        
        if not ground_truth:
            # Without ground truth, cannot calculate recall accurately
            # Return 0.0 or use question as proxy (less accurate)
            return 0.0
        
        # Combine context chunks into single text
        combined_context = " ".join(context)
        
        # Encode combined context and ground truth
        context_embedding = self.model.encode([combined_context], convert_to_numpy=True)
        ground_truth_embedding = self.model.encode([ground_truth], convert_to_numpy=True)
        
        # Calculate similarity (recall: how much of ground truth is covered by context)
        recall_score = float(cosine_similarity(context_embedding, ground_truth_embedding)[0][0])
        
        # Normalize to ensure 0-1 range
        recall_score = max(0.0, min(1.0, recall_score))
        
        return recall_score


# Convenience functions for direct use
def calculate_faithfulness(
    answer: str,
    context: List[str],
    question: str,
    model_name: str = "all-MiniLM-L6-v2"
) -> float:
    """
    Calculate faithfulness score for RAG answer.
    
    Args:
        answer: Generated answer
        context: List of context chunks
        question: Original question
        model_name: Sentence transformer model name
    
    Returns:
        float: Faithfulness score
    """
    evaluator = RAGEvaluator(model_name)
    return evaluator.calculate_faithfulness(answer, context, question)


def calculate_answer_relevancy(
    answer: str,
    question: str,
    model_name: str = "all-MiniLM-L6-v2"
) -> float:
    """
    Calculate answer relevancy score for RAG answer.
    
    Args:
        answer: Generated answer
        question: Original question
        model_name: Sentence transformer model name
    
    Returns:
        float: Answer relevancy score
    """
    evaluator = RAGEvaluator(model_name)
    return evaluator.calculate_answer_relevancy(answer, question)


def calculate_context_precision(
    context: List[str],
    question: str,
    ground_truth: Optional[str] = None,
    model_name: str = "all-MiniLM-L6-v2"
) -> float:
    """
    Calculate context precision for retrieved context.
    
    Args:
        context: List of context chunks
        question: Original question
        ground_truth: Optional ground truth answer
        model_name: Sentence transformer model name
    
    Returns:
        float: Context precision score
    """
    evaluator = RAGEvaluator(model_name)
    return evaluator.calculate_context_precision(context, question, ground_truth)


def calculate_context_recall(
    context: List[str],
    question: str,
    ground_truth: Optional[str] = None,
    model_name: str = "all-MiniLM-L6-v2"
) -> float:
    """
    Calculate context recall for retrieved context.
    
    Args:
        context: List of context chunks
        question: Original question
        ground_truth: Ground truth answer (required for accurate recall)
        model_name: Sentence transformer model name
    
    Returns:
        float: Context recall score
    """
    evaluator = RAGEvaluator(model_name)
    return evaluator.calculate_context_recall(context, question, ground_truth)

