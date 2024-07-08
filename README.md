# Chemical-Gas-Classification-dataset
# Chemical Name Predictor using Machine Learning

## About Dataset

### Dataset Origin
This dataset has been preprocessed from the original dataset collected at a wind tunnel facility designed for chemical detection. It comprises 18,000 time-series recordings from six different locations within the wind tunnel. These recordings were obtained in response to ten high-priority chemical gaseous substances, resulting in a ten-class gas discrimination problem.

🔗For more detailed information, please refer to the [Gas Sensor Arrays in Open Sampling Settings dataset page](https://archive.ics.uci.edu/ml/datasets/gas+sensor+arrays+in+open+sampling+settings) on the UCI Machine Learning Repository.

### Data Collection Details
- **Time Period:** December 2010 to April 2012 (16 months)
- **Location:** Wind tunnel research test-bed facility at the BioCircuits Institute, University of California San Diego

### Dataset Composition
- **Number of Attributes (Features):** Each measurement includes 72 time-series recorded over 260 seconds.
- **Sample Rate:** Data collected at a sample rate of 100 Hz (samples per second).
- **Total Time Series:** The dataset includes 75 time-series, totaling 26,000 points.

### Additional Information
- The dataset includes time, temperature, and relative humidity information, providing contextual details for the chemical detection measurements.
- Prepared for a ten-class gas discrimination problem, making it suitable for classification tasks in machine learning.

## Requirements

### Python Libraries
- numpy
- pandas
- matplotlib
- seaborn
- opendatasets
- scikit-learn

### Installation
You can install the required Python libraries using pip:

```
pip install numpy pandas matplotlib seaborn opendatasets scikit-learn

```


## Conclusion: Analysis of Regression Models


- **Random Forest Classifier**
  - **Training Accuracy**: 100%
  - **Testing Accuracy**: 99.55%
  - **Training Confusion Matrix**: The model perfectly predicted the training data with a normalized confusion matrix showing perfect classification for each chemical class.
  - **Testing Confusion Matrix**: The model also showed very high performance on the test set with a few misclassifications, as reflected in the normalized confusion matrix.

---

- **Feature Importance**
  - The top 10 most important features for the Random Forest Classifier were identified, providing insights into which attributes contributed most to the model's predictions.
  - Detailed feature importance was plotted for all features, highlighting the contribution of each feature in predicting the chemical names.

---

- **Model Performance**
  - The Random Forest Classifier demonstrates excellent performance in predicting chemical names, with an almost perfect training accuracy and very high testing accuracy.
  - Given these results, the Random Forest Classifier is an effective model for predicting the chemical names based on the features provided in the dataset.

### Key Findings

- **High Training Accuracy**: The model achieved a perfect training accuracy, indicating that it learned the patterns in the training data exceptionally well.
- **High Testing Accuracy**: With a **testing accuracy of 99.55%**, the model generalizes well to new, unseen data, making it reliable for practical applications.
- **Feature Importance**: The analysis of feature importance provides valuable insights into which features are most critical for predicting the chemical names, aiding in further refinement and understanding of the model.

## License
[MIT License](LICENSE)
