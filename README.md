# 🎯 Model Evaluation using Logistic Regression and Cross-Validation

This project demonstrates how to evaluate a machine learning model using `LogisticRegression`, `StandardScaler`, and `K-Fold Cross-Validation` in Python. The model predicts biology scores based on math scores.

---

## 📁 Dataset

The code uses a dataset named `firstdataset2.csv` which should contain at least the following columns:
- `math_score`
- `biology_score`

---

## 🧠 What the Code Does

1. **Reads the dataset** using `pandas`
2. **Selects features**: `math_score` as input (`X`) and `biology_score` as target (`Y`)
3. **Creates a pipeline**:
   - `StandardScaler()` for scaling
   - `LogisticRegression()` for classification
4. **Applies 10-fold cross-validation** using `KFold`
5. **Calculates model accuracy** using `cross_val_score`

---

## 📦 Libraries Used

- `pandas`
- `scikit-learn`

Install required libraries with:

```bash
pip install pandas scikit-learn


🚀 How to Run

Make sure your CSV file firstdataset2.csv is in the same folder.

python your_script_name.py

🔍 Output

The script prints the mean accuracy score after 10-fold cross-validation:

mean is 0.87  # (example output)

📌 Key Concepts Practiced

    Logistic Regression

    Pipelines in scikit-learn

    Feature scaling

    Cross-validation (KFold)

    Model evaluation using accuracy

📄 License

MIT License – feel free to use, modify, and share!
🙌 Author

Samson Musyoka
GitHub Profile
⭐️ If you found this useful, give it a star and share!


---

If you'd like me to generate a banner image for your repo or help with uploading `requirements.txt`, just let me know!


