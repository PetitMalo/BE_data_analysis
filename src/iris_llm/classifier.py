import json
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
import ollama
from pydantic import BaseModel, ValidationError
from tqdm import tqdm

from src.iris_llm.dataset import IrisSample

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ClassificationResult(BaseModel):
    """Schema for the LLM's structured output."""

    species: str
    confidence: float
    reasoning: str


class IrisLLMClassifier:
    """Orchestrates the classification of Iris samples using a local LLM via Ollama."""

    def __init__(self, model_name: str = "llama3.2:1b"):
        self.model_name = model_name
        self.system_prompt = (
            "You are a specialized botanical classifier. Your goal is to identify Iris species "
            "with high objectivity. \n\n"
            "### CLASSIFICATION RULES:\n"
            "1. SETOSA: Typically has very small petals (length < 2cm) and wide sepals.\n"
            "2. VERSICOLOR: Medium-sized petals (length between 3cm and 5cm).\n"
            "3. VIRGINICA: Large petals (length > 5cm) and generally larger overall measurements.\n"
            "\n"
            "### CONSTRAINTS:\n"
            "- Do not favor one species over another. Every sample is a new case.\n"
            "- Analyze the ratio between petal length and sepal width carefully.\n"
            "- Respond ONLY in JSON with: 'species', 'confidence', 'reasoning'.\n"
            "- The 'species' value must be lowercase: 'setosa', 'versicolor', or 'virginica'."
        )

    def classify(self, sample: IrisSample) -> ClassificationResult | None:
        """
        Sends a sample to the LLM and parses the structured response.
        """
        prompt = f"Classify this flower: {sample.to_natural_language()}"

        try:
            response = ollama.generate(
                model=self.model_name,
                system=self.system_prompt,
                prompt=prompt,
                format="json",  # Ensures the LLM outputs valid JSON
                options={"temperature": 0},  # Deterministic output for classification
            )

            # Parse the JSON string from the LLM response
            raw_content = response.get("response", "{}")
            data = json.loads(raw_content)

            return ClassificationResult(**data)

        except (json.JSONDecodeError, ValidationError) as e:
            logger.info(f"Error parsing LLM response: {e}")
            return None
        except Exception as e:
            logger.info(f"An unexpected error occurred with Ollama: {e}")
            return None


class ScikitIrisLLM:
    """Wrapper using parallel threads to speed up Ollama inference."""

    def __init__(self, classifier: IrisLLMClassifier, max_workers: int = 4):
        self.classifier = classifier
        self.max_workers = max_workers

    def predict(self, X: List[IrisSample]) -> np.ndarray:
        """Predicts labels for a list of samples using parallel workers."""
        logger.info(f"Inference in progress with {self.max_workers} threads...")

        # We use a thread pool to handle concurrent API calls to Ollama
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # map maintains the order of results, which is crucial for sklearn metrics
            results = list(
                tqdm(
                    executor.map(self.classifier.classify, X),
                    total=len(X),
                    desc="Parallel LLM Inference",
                )
            )

        return np.array([res.species.lower() if res else "unknown" for res in results])


if __name__ == "__main__":
    # Quick integration test
    classifier = IrisLLMClassifier()
    test_sample = IrisSample(
        **{
            "sepal length (cm)": 5.1,
            "sepal width (cm)": 3.5,
            "petal length (cm)": 1.4,
            "petal width (cm)": 0.2,
        }
    )

    result = classifier.classify(test_sample)
    if result:
        logger.info(f"Prediction: {result.species} ({result.confidence * 100}%)")
        logger.info(f"Reasoning: {result.reasoning}")
