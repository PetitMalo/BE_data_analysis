from src.iris_llm.dataset import get_iris_data
from src.iris_llm.classifier import IrisLLMClassifier, ScikitIrisLLM
from src.utils import predict_classifier, metrics_classifier
import numpy as np
from src.iris_llm.classifier import logger


def main():
    samples = get_iris_data(sample=10)
    y_true = np.array([s.target_name.lower() for s in samples])
    
    base_classifier = IrisLLMClassifier(model_name="llama3.2:1b")
    model_wrapper = ScikitIrisLLM(base_classifier)
    
    logger.info("Starting LLM Inference on Iris Dataset...")
    cm = predict_classifier(
        classifier=model_wrapper, 
        x_data=samples, 
        y_true=y_true, 
        plot_cm=True
    )

    # 4. Compute and display metrics
    logger.info("\n--- LLM Performance Report ---")
    metrics = metrics_classifier(cm, plot=True)

    # 5. Final thought
    if metrics['accuracy'] > 0.9:
        logger.info("\nConclusion: The LLM is surprisingly good at botany!")
    else:
        logger.info("\nConclusion: Maybe a 1B model is too small, or the prompt needs tuning.")

if __name__ == "__main__":
    main()