import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

class FakeNewsDetector:
    def __init__(self):
        """
        Initialize the FakeNewsDetector with a TF-IDF vectorizer and a placeholder for the classifier.
        """
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.classifier = MultinomialNB()

    def preprocess_text(self, text):
        """
        Preprocess text by converting it to lowercase and removing special characters and numbers.
        
        :param text: Input text to preprocess.
        :return: Preprocessed text.
        """
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text

    def train_and_evaluate(self, texts, labels):
        """
        Train the Naive Bayes classifier and evaluate its performance.
        
        :param texts: List of input texts.
        :param labels: Corresponding labels for the texts.
        :return: None
        """
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )

        # Transform texts using TF-IDF vectorization
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)

        # Train the Naive Bayes classifier
        self.classifier.fit(X_train_vectorized, y_train)

        # Evaluate the classifier
        y_pred = self.classifier.predict(X_test_vectorized)
        print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred))

    def analyze_dataset(self, df):
        """
        Analyze the dataset and generate predictions with confidence scores.
        
        :param df: Pandas DataFrame containing the dataset.
        :return: DataFrame with analysis results.
        """
        results = []
        for _, row in df.iterrows():
            processed_text = self.preprocess_text(row['text'])
            text_vectorized = self.vectorizer.transform([processed_text])
            prediction = self.classifier.predict(text_vectorized)[0]
            confidence = max(self.classifier.predict_proba(text_vectorized)[0]) * 100

            results.append({
                'text': row['text'],
                'actual_label': row['label'],
                'predicted_label': prediction,
                'confidence': f"{confidence:.2f}%"
            })

        return pd.DataFrame(results)

    def load_and_analyze_csv(self, csv_path):
        """
        Load the dataset, preprocess it, train the classifier, and analyze the data.
        
        :param csv_path: Path to the CSV file containing the dataset.
        :return: DataFrame with analysis results.
        """
        # Load dataset
        df = pd.read_csv(csv_path)

        # Preprocess text data
        df['processed_text'] = df['text'].apply(self.preprocess_text)

        # Train and evaluate the model
        self.train_and_evaluate(df['processed_text'], df['label'])

        # Analyze the dataset
        results_df = self.analyze_dataset(df)
        results_df.to_csv('fake_news_predictions.csv', index=False)

        print("Analysis saved to 'fake_news_predictions.csv'.")
        return results_df

def main():
    detector = FakeNewsDetector()
    results = detector.load_and_analyze_csv('fake-news-dataset.csv')  # Replace with your actual CSV file path
    print(results.head())

if __name__ == "__main__":
    main()
