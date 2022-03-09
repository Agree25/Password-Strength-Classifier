import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import seaborn as sns
import numpy as np
df=pd.read_csv('data.csv',error_bad_lines=False)


print(df['strength'].unique())
print(df.isna().sum())
df.dropna(inplace=True)
print(df.isna().sum())

sns.countplot(x="strength",data=df)
password_tuple=np.array(df)


np.random.shuffle(password_tuple)

x=[labels[0] for labels in password_tuple]
y=[labels[1] for labels in password_tuple]
def word_divide_char(inputs):
    character=[]
    for i in inputs:
        character.append(i)
    return character
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(tokenizer=word_divide_char)

X=vectorizer.fit_transform(x)
vectorizer.get_feature_names()
first_document_vector=X[0]
data=pd.DataFrame(first_document_vector.T.todense(),index=vectorizer.get_feature_names(),columns=['TF-IDF'])
data.sort_values(by=['TF-IDF'],ascending=False) 

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=0,multi_class='multinomial')
clf.fit(X_train,y_train)


dt=np.array(['%@123abcd'])
pred=vectorizer.transform(dt)
clf.predict(pred)
y_pred=clf.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score

cm=confusion_matrix(y_test,y_pred)


from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
print(cm)
print(accuracy_score(y_test,y_pred))
