import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import os
dir = os.path.dirname(os.path.realpath(__file__))  # current directory

# Loading the dataset
df = pd.read_csv(f"{dir}/mail_data.csv")
data = df.where((pd.notnull(df)), '')

data.loc[data['Category'] == 'spam', 'Category'] = 0
data.loc[data['Category'] == 'ham', 'Category'] = 1

# Separating the data
x = data['Message']
y = data['Category']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)


feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
x_train_feature = feature_extraction.fit_transform(x_train)
x_test_feature = feature_extraction.transform(x_test)
y_train = y_train.astype('int')
y_test = y_test.astype('int')

model = LogisticRegression()            #model

model.fit(x_train_feature, y_train)     #training model

y_pred = model.predict(x_test_feature)  #predicting on give data

accuracy = accuracy_score(y_test, y_pred)       #calculating accuracy
print(f'Accuracy: {accuracy}')


#trying custom message
#Example 1 : ham Massage
ham_msg = ['This is the 2nd time we have tried to contact u.']
input_data_feature = feature_extraction.transform(ham_msg)
predicted_class = model.predict(input_data_feature)

if (predicted_class[0]==1):
    print(f"{ham_msg[0]} : is a ham message")
else:
    print(f"{ham_msg[0]} : is a spam message")


#Example 2 Spam Massage
spam_msg = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's."]
input_data_feature = feature_extraction.transform(spam_msg)
predicted_class = model.predict(input_data_feature)

if (predicted_class[0]==1):
    print(f"{spam_msg[0]} : is a ham message")
else:
    print(f"{spam_msg[0]} : is a spam message")


"""
OUTPUT:
Accuracy: 0.9668161434977578
This is the 2nd time we have tried to contact u. : is a ham message
Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's. : is a spam message
"""