import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import svm
from sklearn.metrics import accuracy_score, silhouette_score, davies_bouldin_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import AgglomerativeClustering


def classification():
    df = pd.read_csv("new_data.csv")
    df.drop('1', inplace=True, axis=1)

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


def clustring():
    df = pd.read_csv("leaves.csv", header=None)
    nmi = df[0]
    df.drop(0, inplace=True, axis=1)
    df.drop(1, inplace=True, axis=1)

    whisker_width = 10

    for col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        low_bound = Q1 - whisker_width * IQR
        high_bound = Q3 + whisker_width * IQR
        for i in df.index:
            if df.loc[i, col] < low_bound or df.loc[i, col] > high_bound:
                df.drop(i, inplace=True)
                nmi.drop(i, inplace=True)

    scaler = MinMaxScaler()
    scaler.fit_transform(df)

    model = AgglomerativeClustering(n_clusters=30)
    labels = model.fit_predict(df)

    score = normalized_mutual_info_score(nmi, labels)
    print(f'NMI: {score}')
    score = silhouette_score(df, labels)
    print(f'silhouette score: {score}')
    score = davies_bouldin_score(df, labels)
    print(f'Dunn score: {score}')
