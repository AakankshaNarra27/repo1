
# HR_Analytics_Promotion_Prediction

This repository contains a baseline solution for the HR Analytics promotion prediction task.

Files included:
- `train.csv`, `test.csv`, `sample_submission.csv` — provided data
- `HR_Analytics_Promotion_Prediction.ipynb` — generated notebook with EDA, preprocessing, and model training
- `hr_promotion.py` — runnable script to train and produce `submission_hr_promotion.csv`
- `hr_promotion_pipeline.joblib` — (created after running the notebook/script) saved model pipeline

How to run:
1. Open `HR_Analytics_Promotion_Prediction.ipynb` in Jupyter or Google Colab and run cells sequentially.
2. Or run `python3 hr_promotion.py` (ensure required packages are installed).

Recommended improvements:
- Feature engineering and hyperparameter tuning
- Trying tree boosting models like XGBoost / LightGBM
- Detailed EDA and plotting to explain features
