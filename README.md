1 - Simple version of Credit Card Fraud Detection:
    Random Forest and Logistic Regression is applied.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
(https://colab.research.google.com/github/alimohammadi44/Credit-Card-Fraud-Detection/blob/main/credit_card_fraud_detection_Simple_Case.ipynb)

2- 
| Model                           | Data Split / Validation Strategy | Validation AUC       | Test AUC  | Notes                                                        |
| ------------------------------- | -------------------------------- | -------------------- | --------- | ------------------------------------------------------------ |
| **Random Forest Classifier**    | Train / Test                     | —                    | **0.85**  | Baseline ensemble model                                      |
| **AdaBoost Classifier**         | Train / Test                     | —                    | **0.83**  | Lower performance than Random Forest                         |
| **CatBoost Classifier**         | Train / Test (500 iterations)    | —                    | **0.86**  | Improved performance with boosting                           |
| **XGBoost**                     | Train / Validation / Test        | **0.984**            | **0.974** | Best overall performance; validation used for early stopping |
| **LightGBM**                    | Train / Validation / Test        | **0.974** *(~0.932)* | **0.946** | Strong performance with split validation                     |
| **LightGBM (Cross-Validation)** | Cross-validation + Test          | —                    | **0.93**  | Slightly lower but more robust estimate                      |

Key Takeaways :

The dataset exhibited strong class imbalance, requiring careful evaluation using AUC rather than accuracy.

Tree-based boosting models (XGBoost, LightGBM, CatBoost) consistently outperformed simpler ensemble methods.

XGBoost achieved the highest predictive performance, demonstrating strong generalization.

Cross-validation provided more conservative but robust estimates of model performance.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]
(https://colab.research.google.com/github/alimohammadi44/Credit-Card-Fraud-Detection/blob/main/Credit_Card_Fraud_Detection_Predictive_Models.ipynb)
