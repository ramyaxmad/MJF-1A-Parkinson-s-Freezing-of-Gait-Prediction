# MJF-1A-Parkinson-s-Freezing-of-Gait-Prediction

# Parkinson's Freezing of Gait Prediction

---

### 👥 **Team Members**

| Name             | GitHub Handle | Contribution                                                             |
|------------------|---------------|--------------------------------------------------------------------------|
| Bhuvana Kotha    | @bhuvanak04 | Data exploration, visualization, EDA, RF, RNN            |
| Marc Romero   | @MarcRomero16 | Data visualization, exploratory data analysis (EDA), FFTs, feature engineering, RNN |
| Jeriel Goh    | @JGZH0514  | Data preprocessing, feature engineering, data validation, model training                 |
| Ramya Madugula      | @ramyaxmad       | Data visualization, EDA, FFTs, feature engineering (TSFresh), Tree models |
| Mehek      | @MehekB    | Exploratory data analysis (EDA), Neural networks, Model evaluation, Performance analysis           |
| Saniya      | @chrispark    | Model evaluation, performance analysis, results interpretation           |


---

## 🎯 **Project Highlights**

- Developed a machine learning model using a `supervised classification` approach to detect Freezing of Gait (FoG) in Parkinson's Disease patients from wearable 3D lower back sensor data.
- Project targets `>90% sensitivity and >85% F1 score` to ensure reliability for clinical decision-making.
- The model aims to be `validated, interpretable, and efficient` for potential integration into `wearable devices or mobile health platforms.`
- Generated actionable insights to inform business decisions at `Michael J. Fox Foundation` and to improve patient outcomes.


---

## 👩🏽‍💻 **Setup and Installation**

Our main development workflow used a **Kaggle Notebook**, and we pushed code changes back to this GitHub repo through our individual branches. Below are instructions for reproducing our results on **Kaggle**.

---

### Run on Kaggle

1. **Open the Kaggle notebook**

   - Go to: `Marc’s Kaggle Notebook` {(https://github.com/ramyaxmad/MJF-1A-Parkinson-s-Freezing-of-Gait-Prediction/blob/Marc_Branch/marc-s-notebook.ipynb)}
   - Click **“Copy & Edit”** to save a copy to your own Kaggle account.

2. **Attach the dataset**

   - In the right sidebar, go to **Add data**.
   - Attach the FoG / Parkinson’s competition dataset used in this project 

3. **Check the environment**

   - Ensure the notebook is running a **Python 3** environment with GPU **optional** (CPU is enough, but GPU speeds up training).
   - The notebook installs all required libraries (e.g., `torch`, `polars`, `scipy`, `matplotlib`) via the first setup cells.

4. **Run the notebook**

   - Run all cells from top to bottom:
     - Preprocessing with **Polars**
     - Model definition and training
     - Evaluation (accuracy, precision/recall, ROC curve)
   - At the end, you should see metrics similar to those reported in our presentation.

---

## 🏗️ **Project Overview**

This project is part of the AI Studio Challenge Project, focusing on applying AI/ML to a real-world problem as part of the Break Through Tech AI Program.
The host company is the Michael J. Fox Foundation. The project objective is to develop an ML model, trained on data from a wearable 3D lower back sensor, to detect Parkinson's Freezing of Gait (FoG). The scope is to treat this as a supervised classification task, aiming for high accuracy, sensitivity (>90%), and F1 score (>85%) for potential integration into wearable devices or mobile health platforms.
Freezing of Gait (FoG) is a debilitating symptom of Parkinson's Disease. The successful model will significantly improve patient care by assisting researchers and hospitals, allowing clinicians to identify FoG patterns and adjust treatment plans, which aligns with the MJF Foundation's mission to improve patient outcomes.

---

## 📊 **Data Exploration**

### Dataset

- We worked with a **wearable sensor dataset** for Parkinson’s Disease, containing labeled **Freezing of Gait (FoG)** and **non-FoG** segments.
- **Format:** Multiple CSV files, where each row is a timestamped IMU sample (accelerometer data), along with metadata (subject ID, trial/session) and FoG labels.
- **Structure:**  
  - Columns for time, sensor axes (e.g., acc_ML, acc_AP, acc_V)  
  - A binary label indicating whether each timestep is part of a FoG event or not  
- We loaded and managed these files using **Polars**, which made it easier to handle large time-series tables efficiently.

---

### Data exploration & preprocessing approaches

- **Loading & cleaning (Polars):**
  - Loaded all CSVs into Polars DataFrames.
  - Selected relevant columns (timestamps, IMU signals, labels).
  - Sorted samples by time and grouped by **subject / recording** to reconstruct continuous gait sequences.
  - Checked for missing values, inconsistent dtypes, and dropped or fixed problematic rows if needed.

- **Time-series inspection (Matplotlib):**
  - Plotted **raw accelerometer signals** over time to understand the overall motion patterns.
  - Overlaid **FoG vs non-FoG labels** on the plots (e.g., highlighting FoG segments) to visually compare how the signals change during a freezing episode.
  - Used plots during preprocessing to verify that:
    - Sliding windows had the expected length.
    - Normalization / scaling looked reasonable.
    - Filtering or smoothing did not distort the main movement patterns.

- **Basic statistics & class balance:**
  - Used Polars / NumPy to compute:
    - Mean and standard deviation of each IMU channel.
    - The proportion of **FoG vs non-FoG timesteps** to understand class imbalance.
  - This helped confirm that FoG events are relatively **rare** compared to normal walking, which influenced how we thought about evaluation metrics.

- **Window creation for RNN input:**
  - With Polars, we created **fixed-length sliding windows** for each subject/trial.
  - Each window became one training example:
    - A sequence of IMU features over time.
    - A sequence of labels to train the per-timestep FoG classifier.
  - We checked a few sample windows visually in Matplotlib to make sure:
    - The windows aligned correctly with FoG events.
    - The label sequences matched the plotted signals.

---

### Insights from EDA

- **FoG is sparse and episodic:**  
  FoG labels occur in **short bursts**, surrounded by long stretches of normal walking. This confirmed that:
  - The problem is highly **imbalanced**.
  - Models need to be sensitive to relatively short, abnormal patterns in the signal.

- **Distinct signal patterns near FoG:**  
  By plotting FoG vs non-FoG segments, we observed:
  - Changes in step regularity.
  - Periods where the movement amplitude decreases or becomes irregular.
  These visual patterns motivated using **sequence models (RNNs)** that can track changes over time instead of just single timesteps.

- **Subject variability:**  
  Different subjects showed different overall movement ranges and noise levels, suggesting:
  - The need for **normalization** and robust preprocessing.
  - Caution when interpreting results across subjects.

---

### Challenges & assumptions

- **Label alignment:**  
  We assumed the FoG labels were correctly aligned with the sensor timestamps. Small misalignments could affect per-timestep accuracy, especially near event boundaries.

- **Class imbalance:**  
  Since non-FoG timesteps heavily outnumber FoG timesteps, simple accuracy can be misleading. This is why we also looked at **precision, recall, and ROC/AUC**.

- **Window boundaries:**  
  Some sliding windows may partially overlap FoG events. We assumed our labeling strategy for these mixed windows (e.g., per-timestep labels) was sufficient for the RNN to learn meaningful patterns.

---

### Example visualizations

Some of the key plots we generated during EDA:

- **Time-series plots** of accelerometer/gyroscope data with FoG segments highlighted.
- **Windowed examples** showing how a raw sequence is transformed into an input for the RNN.
- **Metric plots** from evaluation (accuracy, precision/recall, ROC curve) to summarize model performance.

These visualizations helped us understand both the **structure of the data** and what the model is actually learning to detect.
---

## 🧠 **Model Development**

- **Tree Based models**: Decision Tree, XGBoost, and Random Forest. 
- **CNN**: 
- **RNN**: 


---

## 📈 **Results & Key Findings**

- **Tree Based models**: Decision Tree, XGBoost, and Random Forest. The Models appear flawless and raises many concerns on whether the model is able to predict fog based on signal processing data. We can see the metrics - Precision and recall are perfect, roc is perfect. We suspect that because of the way the data we fed into the model has many time windows that overlap and how the labels are based on events, the model sees the same signal repetition. The model is overfitting and not able to handle noise sensitive data. Trees make piecewise predictions, they struggle to model gradual changes in signals. 

- **Random Forest**: 
Accuracy: 88%
ROC-AUC: 0.86
Strong performance on normal gait (99% recall)
Lower recall on rare FoG events (turning and start hesitation)

- **CNN**: 
(Not applicable, trained on wrong data)

- **RNN**: 
Accuracy: 92–93%
Precision: ~0.92
Recall: ~0.89
ROC-AUC: ~0.98

- **LSTM**: 
Accuracy: 91–92%
Precision: ~0.92
Recall: ~0.86
ROC-AUC: ~0.97
---

## 🚀 **Next Steps**

**Model limitations**
- Unable to detect type of freezing of gait when an FOG event is detected by the model
- Overfitting of model, which means model might perform poorly on other types of data. 
  
**What would you do differently with more time/resources?**
- Train CNN model by feeding images of acceleration over time
- Delve into Deep learning models by feeding spectrogram data

**What additional datasets or techniques would you explore?**
- Reduce dimensionality technique
- Using shap to determine which features are most significant.


---

## 📝 **License**

This project is licensed under the MIT License.

---

## 📄 **References** 

Cite relevant papers, articles, or resources that supported your project.

Manor B;Dagan M;Herman T;Gouskova NA;Vanderhorst VG;Giladi N;Travison TG;Pascual-Leone A;Lipsitz LA;Hausdorff JM; (n.d.). Multitarget transcranial electrical stimulation for freezing of gait: A randomized controlled trial. Movement disorders : official journal of the Movement Disorder Society. https://pubmed.ncbi.nlm.nih.gov/34406695/ 

Reches T;Dagan M;Herman T;Gazit E;Gouskova NA;Giladi N;Manor B;Hausdorff JM; (n.d.). Using wearable sensors and machine learning to automatically detect freezing of gait during a fog-provoking test. Sensors (Basel, Switzerland). https://pubmed.ncbi.nlm.nih.gov/32785163/ 

Salomon, A., Gazit, E., Ginis, P., Urazalinov, B., Takoi, H., Yamaguchi, T., Goda, S., Lander, D., Lacombe, J., Sinha, A. K., Nieuwboer, A., Kirsch, L. C., Holbrook, R., Manor, B., & Hausdorff, J. M. (2024, June 6). A machine learning contest enhances automated freezing of gait detection and reveals time-of-day effects. Nature News. https://www.nature.com/articles/s41467-024-49027-0 

---

## 🙏 **Acknowledgements** 

We would like to extend a huge thank you to Harshini, Seth, and Barbara for all their time, expert guidance, and immense encouragement throughout the project. Thank you so much for helping us achieve this milestone!

