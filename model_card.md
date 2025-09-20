# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

Model type: Logistic Regression (sklearn.linear_model.LogisticRegression)

Implementation: scikit-learn

Features: Combination of numeric (e.g., age, hours-per-week) and categorical (e.g., workclass, education, marital-status, occupation, relationship, race, sex, native-country) features.

Preprocessing: One-hot encoding for categorical variables, label binarization for the target.

## Intended Use

Primary use: Educational / demonstration purposes for learning about deploying ML pipelines with FastAPI.

Not intended for: Real-world decision-making about income prediction or employment screening.

## Training Data

Source: data/census.csv (UCI Census Income dataset).

Size: ~32,000 rows, with ~108 processed features after encoding.

Label: salary (binary: <=50K vs >50K).

## Evaluation Data

Split: 80% training, 20% test.

Stratified by the label to balance class distribution.

## Metrics

Overall Performance on Test Set:

Precision: 0.7205

Recall: 0.6084

F1: 0.6598

Per-slice Performance: Recorded in slice_output.txt.

Example: On sex=Female, F1 was lower than overall average.

Example: On race=Black, recall dropped compared to the overall dataset.

These variations highlight fairness and bias considerations.

## Ethical Considerations

The model is trained on census data which reflects real-world social and economic inequalities.

Predictions may perpetuate bias present in the data (e.g., across gender, race, or country of origin).

Must not be used for employment or lending decisions.

## Caveats and Recommendations

Logistic Regression is a simple baseline; more advanced models (Random Forest, XGBoost) could yield higher performance.

Model performance varies across slices (see slice_output.txt), suggesting potential bias.

Future work: mitigate bias with fairness-aware preprocessing, tune hyperparameters, and test robustness with cross-validation.

Recommendation: always pair quantitative metrics with qualitative review when using demographic data.
