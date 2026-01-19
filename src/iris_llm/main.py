import numpy as np

from src.iris_llm.classifier import IrisLLMClassifier, ScikitIrisLLM, logger
from src.iris_llm.dataset import get_iris_data
from src.utils import metrics_classifier, predict_classifier


def main(num_sample: int = 10):
    samples = get_iris_data(num_sample=num_sample)
    y_true = np.array([s.target_name.lower() for s in samples])

    base_classifier = IrisLLMClassifier(model_name="llama3.2:1b")
    model_wrapper = ScikitIrisLLM(base_classifier)

    logger.info("Starting LLM Inference on Iris Dataset...")
    cm = predict_classifier(classifier=model_wrapper, x_data=samples, y_true=y_true, plot_cm=True)

    logger.info("--- LLM Performance Report ---")
    metrics = metrics_classifier(cm, plot=True)

    if metrics["accuracy"] > 0.9:
        logger.info("\nConclusion: The LLM is surprisingly good at botany!")
    else:
        logger.info("\nConclusion: Maybe a 1B model is too small, or the prompt needs tuning.")

    return metrics


if __name__ == "__main__":
    main()
