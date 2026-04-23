# Zero Frequency Problem and Laplace Smoothing in Naive Bayes

# Instructions: 
# 1. The code creates a simple dataset for Naive Bayes classification.
# 2. It separates the input features and the target label.
# 3. The dataset is used to train a Naive Bayes model twice:
#       First without smoothing (alpha = 0)
#       Then with Laplace smoothing (alpha = 1)
# 4. The model is tested using a sample input transaction.
# 5. The program prints the prediction results for both models.
# 6. The output shows how smoothing affects the prediction results.
# 7. Compare both outputs to understand the effect of the zero frequency problem.

# Import libraries
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

# Sample data
data = {
    'International': [1, 1, 1, 1, 0, 0],
    'Fraud': [1, 1, 1, 1, 0, 0]
}

df = pd.DataFrame(data)

X = df[['International']]
y = df['Fraud']

test_case = [[1]]  # Test input

# No smoothing
model1 = MultinomialNB(alpha=0)
model1.fit(X, y)
print("No smoothing:", model1.predict(test_case))

# With smoothing
model2 = MultinomialNB(alpha=1)
model2.fit(X, y)
print("With smoothing:", model2.predict(test_case))