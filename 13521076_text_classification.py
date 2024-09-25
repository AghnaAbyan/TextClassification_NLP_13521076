import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Mengunduh stopwords Bahasa Indonesia
nltk.download('stopwords')
list_stopwords = set(stopwords.words('indonesian'))

# Preprocessing
def preprocess(text):
    # 1. Ubah teks menjadi huruf kecil
    # 2. Menghilangkan tanda baca, simbol, dan angka
    # 3. Proses tokenization
    # 4. Hilangkan semua stopwords
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = text.split()
    text = [word for word in text if word not in list_stopwords]
    return ' '.join(text)

# Memuat dataset serta melakukan preprocessing
train_data = pd.read_csv('dataset/smsa_doc-sentiment-prosa/train_preprocess.tsv', sep='\t')
train_data = train_data.rename(columns={train_data.columns[0]: 'sentence', train_data.columns[1]: 'label'})
train_data['preprocessing_sentence'] = train_data['sentence'].apply(preprocess)

valid_data = pd.read_csv('dataset/smsa_doc-sentiment-prosa/valid_preprocess.tsv', sep='\t')
valid_data = valid_data.rename(columns={valid_data.columns[0]: 'sentence', valid_data.columns[1]: 'label'})
valid_data['preprocessing_sentence'] = valid_data['sentence'].apply(preprocess)

# Inisialisasi CountVectorizer untuk Bag of Words
vectorizer = CountVectorizer()

# Ambil kalimat yang sudah melalui proses preprocessing
X_train = vectorizer.fit_transform(train_data['preprocessing_sentence'])
X_valid = vectorizer.transform(valid_data['preprocessing_sentence'])
y_train = train_data['label']
y_valid = valid_data['label']

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_valid)

# SVM
svm_model = SVC()
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_valid)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_valid)

# Fungsi evaluasi model
def evaluate_model(a, b):
    accuracy = accuracy_score(a, b)
    precision = precision_score(a, b, average='weighted')
    recall = recall_score(a, b, average='weighted')
    f1 = f1_score(a, b, average='weighted')
    return accuracy, precision, recall, f1

# Evaluasi tiap model
acc_lr, pre_lr, rec_lr, f1_lr = evaluate_model(y_valid, y_pred_lr)
print(f"""1) Logistic Regression
      Accuracy: {acc_lr}, 
      Precision: {pre_lr}, 
      Recall: {rec_lr}, 
      F1-Score: {f1_lr}""")

acc_svm, pre_svm, rec_svm, f1_svm = evaluate_model(y_valid, y_pred_svm)
print(f"""2) SVM
      Accuracy: {acc_svm}, 
      Precision: {pre_svm}, 
      Recall: {rec_svm}, 
      F1-Score: {f1_svm}""")

acc_rf, pre_rf, rec_rf, f1_rf = evaluate_model(y_valid, y_pred_rf)
print(f"""3) Random Forest
      Accuracy: {acc_rf}, 
      Precision: {pre_rf}, 
      Recall: {rec_rf}, 
      F1-Score: {f1_rf}""")