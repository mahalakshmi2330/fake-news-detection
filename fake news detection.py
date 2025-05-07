import pandas as pd
import nltk
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
nltk.download('stopwords')
from nltk.corpus import stopwords
# Sample data (you should load actual Reddit data with 'text' and 'label' columns)
# For example: label = 1 for fake, 0 for real
df = pd.read_excel('C:\\Users\\maham\\Downloads\\fake_news_dataset.xlsx')
# Preprocessing
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)
df['text'] = df['text'].apply(clean_text)

# Split
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.25, random_state=42)

# Vectorize
vectorizer = TfidfVectorizer()
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# Model
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(x_train_vec, y_train)

# Predict
y_pred = model.predict(x_test_vec)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


