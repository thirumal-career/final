import pandas as pd
import unicodedata
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize,pos_tag
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from collections import Counter
from nltk.corpus import stopwords
from sklearn.metrics import classification_report
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import nltk
#nltk.download()
nltk.data.path.append('/home/pp/nltk_data/tokenizers/punkt')
nltk.data.path.append('/home/pp/nltk_data/corpora/stopwords')
tt = pd.read_csv("twitter_training.csv",header=None)
cn=["id","tid","target","sentiment"]
tt.columns = cn
tt = tt[tt['sentiment'].notna()]
mappings = {'Irrelevant': 'Neutral'}
tt['target'] = tt['target'].replace(mappings)
tt.drop(columns=['id','tid'],inplace=True)
def emojihint(texts):
    for text in texts:
        for char in text:
            if char != ' ' and unicodedata.name(char).startswith('EMOJI'):
                return True
    return False
tt['emoji'] = tt['sentiment'].apply(emojihint)
tt = tt[tt['emoji'] == False ]
tt['sentiment']=tt['sentiment'].str.lower()
def remove_double_space(text):
    return  " ".join(text.split())
tt['sentiment']=tt['sentiment'].apply(remove_double_space)

tt['sentiment']=tt['sentiment'].apply(lambda X: word_tokenize(X))

en_stopwords = stopwords.words('english')
def remove_stopwords(text):
    en_stopwords = stopwords.words('english')
    negations = ["not", "no", "never", "n't"]
    result = [token for token in text if token.lower() not in en_stopwords or token.lower() in negations]
    return result
tt['sentiment']=tt['sentiment'].apply(remove_stopwords)

def remove_punct(text):

    tokenizer = RegexpTokenizer(r"\w+")
    lst=tokenizer.tokenize(' '.join(text))
    return lst


tt['sentiment']=tt['sentiment'].apply(remove_punct)


def keep_alphabetical_only(sentiment_list):
    return [word for word in sentiment_list if word.isalpha()]

# Apply the function to the 'sentiment' column
tt['sentiment'] = tt['sentiment'].apply(keep_alphabetical_only)


def remove_im(sentiment_list):
    return [word for word in sentiment_list if word != 'im']

# Apply the function to the 'sentiment' column
tt['sentiment'] = tt['sentiment'].apply(remove_im)

# Convert lists to tuples in the 'sentiment' column
tt['sentiment'] = tt['sentiment'].apply(tuple)

# Drop duplicates
tt = tt.drop_duplicates()

# Convert tuples back to lists (if necessary)
tt['sentiment'] = tt['sentiment'].apply(list)


def remove_single_letters(text):
    return [word for word in text if len(word) > 1]

# Apply the function to the 'sentiment' column
tt['sentiment'] = tt['sentiment'].apply(remove_single_letters)
     

tt["len_sent"] = tt['sentiment'].apply(lambda X: len(X))

tt = tt[tt['len_sent'] != 0]

def lemmatization(text):

    result=[]
    wordnet = WordNetLemmatizer()
    for token,tag in pos_tag(text):
        pos=tag[0].lower()

        if pos not in ['a', 'r', 'n', 'v']:
            pos='n'

        result.append(wordnet.lemmatize(token,pos))

    return result


tt['sentiment']=tt['sentiment'].apply(lemmatization)


tt.drop(columns=["emoji","len_sent"],inplace=True)
     
tt['sentiment_joined'] = tt['sentiment'].apply(lambda x: ' '.join(x))


chat_words_str = """
AFAIK=As Far As I Know
AFK=Away From Keyboard
ASAP=As Soon As Possible
ATK=At The Keyboard
ATM=At The Moment
A3=Anytime, Anywhere, Anyplace
BAK=Back At Keyboard
BBL=Be Back Later
BBS=Be Back Soon
BFN=Bye For Now
B4N=Bye For Now
BRB=Be Right Back
BRT=Be Right There
BTW=By The Way
B4=Before
B4N=Bye For Now
CU=See You
CUL8R=See You Later
CYA=See You
FAQ=Frequently Asked Questions
FC=Fingers Crossed
FWIW=For What It's Worth
FYI=For Your Information
GAL=Get A Life
GG=Good Game
GN=Good Night
GMTA=Great Minds Think Alike
GR8=Great!
G9=Genius
IC=I See
ICQ=I Seek you (also a chat program)
ILU=ILU: I Love You
IMHO=In My Honest/Humble Opinion
IMO=In My Opinion
IOW=In Other Words
IRL=In Real Life
KISS=Keep It Simple, Stupid
LDR=Long Distance Relationship
LMAO=Laugh My A.. Off
LOL=Laughing Out Loud
LTNS=Long Time No See
L8R=Later
MTE=My Thoughts Exactly
M8=Mate
NRN=No Reply Necessary
OIC=Oh I See
PITA=Pain In The A..
PRT=Party
PRW=Parents Are Watching
ROFL=Rolling On The Floor Laughing
ROFLOL=Rolling On The Floor Laughing Out Loud
ROTFLMAO=Rolling On The Floor Laughing My A.. Off
SK8=Skate
STATS=Your sex and age
ASL=Age, Sex, Location
THX=Thank You
TTFN=Ta-Ta For Now!
TTYL=Talk To You Later
U=You
U2=You Too
U4E=Yours For Ever
WB=Welcome Back
WTF=What The F...
WTG=Way To Go!
WUF=Where Are You From?
W8=Wait...
7K=Sick:-D Laugher
"""


chat_words_dict = dict(line.split('=') for line in chat_words_str.strip().split('\n'))


# Add a text length column
#tt['text_length'] = tt['sentiment_joined'].apply(len)

# Display the text length statistics
#print(tt['text_length'].describe())


#sns.histplot(df['text_length'], kde=True)
#plt.title('Distribution of Text Length')
#plt.xlabel('Text Length')
#plt.ylabel('Frequency')
#plt.show()

# Combine all text for positive sentiment
#positive_texts = ' '.join(tt[tt['target'] == 'Positive']['sentiment_joined'])

# Generate word cloud
#wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_texts)

# Plot word cloud
#plt.figure(figsize=(10, 5))
#plt.imshow(wordcloud_pos, interpolation='bilinear')
#plt.axis('off')
#plt.title('Word Cloud for Positive Sentiment')
#plt.show()


# Combine all text for positive sentiment
#positive_texts = ' '.join(tt[tt['target'] == 'Negative']['sentiment_joined'])

# Generate word cloud
#wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_texts)

# Plot word cloud
#plt.figure(figsize=(10, 5))
#plt.imshow(wordcloud_pos, interpolation='bilinear')
#plt.axis('off')
#plt.title('Word Cloud for Negative Sentiment')
#plt.show()

#positive_texts = ' '.join(tt[tt['target'] == 'Neutral']['sentiment_joined'])

# Generate word cloud
#wordcloud_pos = WordCloud(width=800, height=400, background_color='white').generate(positive_texts)

# Plot word cloud
#plt.figure(figsize=(10, 5))
#plt.imshow(wordcloud_pos, interpolation='bilinear')
#plt.axis('off')
#plt.title('Word Cloud for Neutral Sentiment')
#plt.show()

#sns.boxplot(x='target', y='text_length', data=df)
#plt.title('Text Length Distribution by Sentiment')
#plt.xlabel('Sentiment')
#plt.ylabel('Text Length')
#plt.show()

# Function to check for chat words
def contains_chat_word(text):
    words = text.split()
    for word in words:
        if word in chat_words_dict:
            return True
    return False

# Apply the function to the DataFrame
tt['contains_chat_word'] = tt['sentiment_joined'].apply(contains_chat_word)

tt['text_length'] = tt['sentiment_joined'].apply(len)


# Function to get most common words
def get_most_common_words(text, n=10):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalpha()]  # keep only alphabetic words
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # remove stopwords
    return Counter(tokens).most_common(n)

# Combine text by sentiment
positive_text = ' '.join(tt[tt['target'] == 'Positive']['sentiment_joined'])

# Get most common words
common_words_pos = get_most_common_words(positive_text, 10)


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(tt['sentiment_joined'])
     

# Convert target column to numerical values
le = LabelEncoder()
y = le.fit_transform(tt['target'])
     

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


model = RandomForestClassifier()

# Train the ExtraTreesRegressor
model.fit(X_train, y_train)

# Predict on the test set
#y_pred = model.predict(X_test)

#y_pred_train = model.predict(X_train)

#r2_test = accuracy_score(y_test, y_pred)

#r2_train = accuracy_score(y_train, y_pred_train)


#a = "i have RE bike in my carrage "
#a = a.lower()
#a = " ".join(a.split())
#a = word_tokenize(a)
#a = remove_stopwords(a)
#a = remove_punct(a)
#a = keep_alphabetical_only(a)
#a = remove_single_letters(a)
#a = lemmatization(a)
#a = " ".join(a)

#aa_transformed = vectorizer.transform([a])

# Predict the sentiment using the loaded model
#y_pred = model.predict(aa_transformed)




with open('rf.pkl', 'wb') as file:
   pickle.dump(model, file)
     







































