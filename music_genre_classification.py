

from sklearn.model_selection import train_test_split, KFold, cross_validate
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU,Input
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score, classification_report
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix


warnings.filterwarnings('ignore')
df = pd.read_csv("C:\\Users\\Ebra Nur Sayar\\Desktop\\train.csv")

print(df.head())
print(df.shape)
print(df.info())

missing_values = df.isnull().sum()

class_distribution = df['Class'].value_counts()

print(missing_values, class_distribution)

df['Popularity'].fillna(df['Popularity'].median(), inplace=True)
df['key'].fillna(df['key'].mode()[0], inplace=True)
df['instrumentalness'].fillna(0, inplace=True)


data_cleaned1 = df.drop(columns=['Artist Name', 'Track Name'])


print(data_cleaned1.isnull().sum())
print(data_cleaned1.info)

from sklearn.preprocessing import MinMaxScaler, LabelBinarizer


scaler = MinMaxScaler()
numerical_features = data_cleaned1.columns.difference(['Class'])

data_cleaned1[numerical_features] = scaler.fit_transform(data_cleaned1[numerical_features])

data_cleaned1.head()
print(data_cleaned1.info())

numerical_features = data_cleaned1.columns.difference(['Class'])

plt.figure(figsize=(15, 10))
for i, column in enumerate(numerical_features, 1):
    plt.subplot(4, 4, i)
    sns.boxplot(data_cleaned1[column])
    plt.title(column)

plt.tight_layout()
plt.show()

def outlier(df):
    q1=df.quantile(0.25)
    q3=df.quantile(0.75)
    iqr=q3-q1
    lower=q1-1.5*iqr
    upper=q3+1.5*iqr
    return lower,upper
def replace_thresh(df,col):
    lower, upper = outlier(df[col])
    df[col] = df[col].apply(lambda x: lower if x < lower else (upper if x > upper else x))
    return df
for col in data_cleaned1.columns:
    replace_thresh(data_cleaned1,col)

X = data_cleaned1.drop(columns=['Class'])
y = data_cleaned1['Class']

print("Eşsiz sınıf değerleri:", np.unique(y))

num_classes = y.max() + 1
y = to_categorical(y, num_classes=num_classes)
print("X shape:", X.shape)
print("y shape:", y.shape)
print(data_cleaned1.shape)
print(data_cleaned1.tail(50))

def create_model(input_dim,learning_rate):
    model = Sequential([

        Input(shape=(input_dim,)),
        Dense(512,'relu'),  
        Dropout(0.2),
        Dense(256, 'relu'),
        #Dropout(0.2),
        Dense(128,'relu'),
        #Dropout(0.2),
        Dense(64,'relu'),
        #Dropout(0.2),
        Dense(32,'relu'),
        Dense(16,'relu'),
        Dense(11,'softmax')

    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True)
    return model
input_dim=X.shape[1]


model.fit(X, y, epochs=100, batch_size=32, verbose=1)  # Eğitim

y_pred = np.argmax(model.predict(X), axis=1)

y_true = np.argmax(y, axis=1)

print(classification_report(y_true, y_pred))



print("%66-%34 Eğitim Test Ayrımı (5 Farklı Rassal Ayırma):")
random_accuracies = []
random_f1_scores = []
for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=i)
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    random_accuracies.append(accuracy)


    y_pred = model.predict(X_test, batch_size=16, verbose=0)
    y_pred_classes = np.argmax(y_pred, axis=1)

    y_test_classes = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test


    f1 = f1_score(y_test_classes, y_pred_classes,
                  average='weighted')
    random_f1_scores.append(f1)


    print(f"Split {i + 1} - Accuracy: {accuracy * 100:.2f}%, F1 Score: {f1:.4f}")



avg_accuracy = np.mean(random_accuracies) * 100
avg_f1 = np.mean(random_f1_scores)
print(f"Average Accuracy (%66-%34 Split): {avg_accuracy:.2f}%")
print(f"Average F1 Score (%66-%34 Split): {avg_f1:.4f}")

#5V

X = X.values
y = y if isinstance(y, np.ndarray) else y.values
def train_with_kfold_5(model, X, y):
    print("5-Fold Cross Validation:")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_f1_scores = []


    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = create_model(input_dim=X.shape[1], learning_rate=0.001)


        model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)


        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
        fold_accuracies.append(accuracy)

        y_pred = model.predict(X_test, batch_size=32, verbose=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        f1 = f1_score(y_pred_classes, y_test_classes, average='weighted')
        fold_f1_scores.append(f1)

        print(f"Fold Accuracy: {accuracy * 100:.2f}%, Fold F1 Score: {f1*100:.2f}")

    avg_accuracy = np.mean(fold_accuracies) * 100
    avg_f1 = np.mean(fold_f1_scores)
    print(f"Average Accuracy (5-Fold): {avg_accuracy:.2f}%")
    print(f"Average F1 Score (5-Fold): {avg_f1:.2f}")
    return y_pred_classes,y_test_classes

y_pred_classes,y_test_classses=train_with_kfold_5(lambda: create_model(input_dim=X.shape[1], learning_rate=0.001), X, y)


def train_with_kfold_10(model, X, y):
    print("10-Fold Cross Validation:")
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_accuracies = []
    fold_f1_scores = []

    for train_idx, test_idx in kfold.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        model = create_model(input_dim=X.shape[1], learning_rate=0.001)


        model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1)


        loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
        fold_accuracies.append(accuracy)


        y_pred = model.predict(X_test, batch_size=32, verbose=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)

        f1 = f1_score(y_pred_classes, y_test_classes, average='weighted')
        fold_f1_scores.append(f1)

        print(f"Fold Accuracy: {accuracy * 100:.2f}%, Fold F1 Score: {f1*100:.2f}")

    avg_accuracy = np.mean(fold_accuracies) * 100
    avg_f1 = np.mean(fold_f1_scores)
    print(f"Average Accuracy (10-Fold): {avg_accuracy:.2f}%")
    print(f"Average F1 Score (10-Fold): {avg_f1:.2f}")
    return y_pred_classes,y_test_classes


y_pred_classes,y_test_classses=train_with_kfold_10(lambda: create_model(input_dim=X.shape[1], learning_rate=0.001), X, y)



def plot_confusion_matrix(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix", fontsize=16)
    plt.xlabel("Predicted Labels", fontsize=14)
    plt.ylabel("True Labels", fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    return cm
plot_confusion_matrix(y_pred_classes,y_test_classses)