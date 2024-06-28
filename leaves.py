import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

df = pd.read_csv("new_data.csv")
df.drop('1', inplace=True, axis=1)


def classification():
    train, test = train_test_split(df, test_size=0.2, random_state=200)
    test_y = test['0']
    test.drop('0', axis=1, inplace=True)

    whisker_width = 10

    for col in train.columns:
        if col == '0':
            continue
        Q1 = train[col].quantile(0.25)
        Q3 = train[col].quantile(0.75)
        IQR = Q3 - Q1
        low_bound = Q1 - whisker_width * IQR
        high_bound = Q3 + whisker_width * IQR
        for i in train.index:
            if train.loc[i, col] < low_bound or train.loc[i, col] > high_bound:
                train.drop(i, inplace=True)

    train_y = train['0']
    train.drop('0', axis=1, inplace=True)

    scaler = StandardScaler()
    scaler.fit(train)

    train = scaler.transform(train)
    test = scaler.transform(test)

    train = pd.DataFrame(train)
    test = pd.DataFrame(test)

    clf = svm.SVC(C=1, kernel='linear')
    clf.fit(train, train_y)

    predictions = clf.predict(test)

    accuracy = accuracy_score(test_y, predictions)
    print("Accuracy:", accuracy)
