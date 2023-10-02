import nltk
#import textblob
from textblob import TextBlob
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.sentiment import SentimentIntensityAnalyzer
import string

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Sample open-ended text
text = """
Analyzing open-ended text can be a complex task, but it's essential for extracting valuable insights. 
This script demonstrates basic text analysis using Python's NLTK and TextBlob libraries. 
We'll perform tokenization, stop word removal, sentiment analysis, and frequency analysis.
"""

# Tokenization and sentence splitting
tokens = word_tokenize(text)
sentences = sent_tokenize(text)

# Removing stopwords and punctuation
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word not in string.punctuation]

# Frequency distribution
freq_dist = FreqDist(filtered_tokens)
print("Top 10 most common words:")
print(freq_dist.most_common(10))

# Sentiment analysis using TextBlob
blob = TextBlob(text)
sentiment = blob.sentiment
print("\nSentiment Analysis (using TextBlob):")
print(f"Polarity: {sentiment.polarity}")
print(f"Subjectivity: {sentiment.subjectivity}")

# Sentiment analysis using VADER (NLTK)
sia = SentimentIntensityAnalyzer()
sentiment_scores = sia.polarity_scores(text)
print("\nSentiment Analysis (using VADER):")
print(f"Positive: {sentiment_scores['pos']}")
print(f"Neutral: {sentiment_scores['neu']}")
print(f"Negative: {sentiment_scores['neg']}")

# Print tokenized sentences
print("\nTokenized Sentences:")
for i, sentence in enumerate(sentences, start=1):
    print(f"Sentence {i}: {sentence.strip()}")

# Print filtered tokens (without stopwords and punctuation)
print("\nFiltered Tokens:")
print(filtered_tokens)
