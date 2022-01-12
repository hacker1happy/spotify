import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import covariance
from sklearn import model_selection
import pickle

df = pd.read_csv("genre.csv")
df.drop(columns=['type', 'id', 'uri', 'track_href', 'analysis_url', 'song_name'], inplace=True)


### outlayers - IQR
def IQR_outliers(data, outlayers):
  Q1, Q3 = np.percentile(data, [25, 75])
  IQR = Q3 - Q1

  u_band  = Q3 + (1.5 * IQR)
  l_band  = Q1 - (1.5 * IQR)
  idx = np.where((data > u_band) | (data < l_band))
  outlayers[idx] = True
  
  return outlayers

#outlayers array, by default row is treated as non outlayer
outlayers = np.full(shape=df.shape[0], fill_value=False)

cols = df.select_dtypes(include=np.number).columns.tolist()
#for each feature - mark outlayers
for col in cols[:-1]:
  outlayers = IQR_outliers(df[col], outlayers)


outIdx = [i for i,x in enumerate(outlayers) if x == True]
df_nonoutlayers_by_iqr = df.drop(outIdx)
df_nonoutlayers_by_iqr.head(10)
df_nonoutlayers_by_iqr.shape


### Outlayers - elipticEnvelope


detector = covariance.EllipticEnvelope(contamination=0.1, support_fraction=1)

toOutlayers = df.drop(columns="genre")
detector.fit(toOutlayers)

ol_flag = detector.predict(toOutlayers)

outIdx = [i for i,x in enumerate(ol_flag) if x == -1]

df_nonoutlayers_by_envelope = df.drop(outIdx)
df_nonoutlayers_by_envelope.head(10)
df_nonoutlayers_by_envelope.shape 

df_pre_model = df.copy()

low_var = df_pre_model.var()
low_var = low_var[low_var < 0.02 ]

low_var_cols = low_var.index.tolist()
df_pre_model.drop(columns=low_var_cols, inplace=True);


df_category = pd.get_dummies(df_pre_model, columns=['time_signature'], prefix_sep='_')


##Data Modeling
X1 = df_category.drop(columns=['genre'])
Y1 = df_category["genre"]
X1.head()

from imblearn.over_sampling import SMOTE

X1_std = preprocessing.StandardScaler().fit_transform(X1)


smote = SMOTE()
X1, Y1 = smote.fit_resample(X1_std, Y1)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X1, 
                                                                    Y1, 
                                                                    test_size=.2, 
                                                                    random_state=1,
                                                                    shuffle=True
                                                                    )



from sklearn.ensemble import BaggingClassifier
# BaggingClassifier

algBag = BaggingClassifier()
algBag.fit(X_train, y_train)
predict = algBag.predict(X_test)

# Saving model to disk
pickle.dump(algBag, open('model.pkl','wb'))

# # Loading model to compare the results
# model = pickle.load(open('model.pkl','rb'))

# x = [[0.719, 0.493, 8, -7.23, 1, 0.401, 0, 0.118, 0.124, 115.08, 224427, 0, 0, 1, 0]]
# print(algBag.predict(x)[0])
