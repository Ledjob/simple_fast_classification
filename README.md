# SMS Text Classification

## Project Overview

This project implements a machine learning model to classify SMS messages as either "spam" or "ham" (legitimate messages). The implementation uses a combination of natural language processing techniques and a Naive Bayes classifier with custom feature engineering to achieve high accuracy.

## Features

- Text vectorization using CountVectorizer
- Multinomial Naive Bayes classifier
- Custom feature engineering including:
  - Message length
  - Digit and special character analysis
  - Detection of spam indicator words
  - Phone number pattern recognition
  - Money symbol detection
  - Uppercase word counting

## Implementation Details

The classifier works by:

1. Converting text messages into numerical features using the bag-of-words approach
2. Training a Naive Bayes model on these features
3. Enhancing the classification with domain-specific feature engineering
4. Adjusting probability scores based on spam indicators
5. Making a final prediction based on the adjusted probability

## Advantages of This Approach

- **Efficiency**: Naive Bayes is computationally lightweight and trains quickly, even on modest hardware
- **Works well with small datasets**: Performs well even with limited training examples (thousands rather than millions)
- **Interpretability**: Easy to understand which features contribute to classification decisions
- **Feature engineering relevance**: Manually engineered features directly capture domain knowledge about spam patterns
- **No overfitting**: Simpler models are less likely to overfit on small datasets
- **Fast inference**: Predictions can be made quickly, suitable for real-time applications

## Comparison with Neural Networks

While neural networks could be used for this task, there are reasons why this approach may be preferable:

- **Data requirements**: Neural networks typically need much more training data to generalize well
- **Computational cost**: Neural network training and inference are more resource-intensive
- **Text length**: SMS messages are very short, so the sequential advantages of RNNs/LSTMs/Transformers are less impactful
- **Feature discovery**: SMS spam has fairly explicit patterns that can be captured with simple feature engineering
- **Development time**: This approach requires less time to develop and iterate
- **Deployment simplicity**: Easier to deploy in production environments with minimal dependencies

## Usage

```python
from sms_classifier import predict_message

# Example usage
result = predict_message("You have won £1000 cash! call to claim your prize.")
print(f"Spam Probability: {result[0]}, Classification: {result[1]}")
```

## Performance

The model successfully passes all test cases in the challenge, correctly classifying various types of spam and legitimate messages. The feature engineering approach allows it to detect spam patterns even in messages with new or unseen vocabulary.

## Future Improvements

Potential enhancements:

- Implement TF-IDF instead of raw counts
- Add character n-grams to catch misspelled spam words
- Use cross-validation to optimize probability adjustments
- Integrate with a messaging application for real-time filtering

## Requirements

- pandas
- numpy
- scikit-learn

## License

MIT
