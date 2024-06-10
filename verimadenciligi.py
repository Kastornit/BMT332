""" KÜTÜPHANELERİN IMPORT EDİLMESİ """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

""" VERİLERİN IMPORT EDİLMESİ """

veriseti = pd.read_csv('name_gender_dataset.csv')
veriseti.head(10)


""" VERİ SETİ HAKKINDA """


pd.set_option('display.float_format', '{:0.8f}'.format)
#Bilimsel notasyonu kapattık

veriseti.describe()


veriseti.info()

""" VERİLERİN GÖRSELLEŞTİRİLMESİ """

#ÇUBUK GRAFİĞİ İLE CİNSİYET DAĞILIMI 
cinsiyet_sayac = veriseti['Gender'].value_counts()
plt.bar(cinsiyet_sayac.index, cinsiyet_sayac.values)

plt.xlabel("Cinsiyet")
plt.ylabel("Adet")
plt.title("Cinsiyet Dağılımı")

plt.show()

#PASTA GRAFİĞİ İLE CİNSİYET DAĞILIMI 
cinsiyet_sayac = veriseti['Gender'].value_counts()
plt.pie(cinsiyet_sayac, labels=cinsiyet_sayac.index, autopct='%1.1f%%', startangle=140)

plt.title("Cinsiyet Dağılımı")

plt.axis()
plt.show()

#EN POPÜLER 7 İSİM

populer_isimler = veriseti.sort_values(by=['Count'], ascending=False)

en_populer_7_isim = populer_isimler.head(7)

renkler = plt.cm.Paired(range(7))

plt.bar(en_populer_7_isim['Name'], en_populer_7_isim['Count'], color=renkler)
plt.xlabel('İsimler')
plt.ylabel('Sayı')
plt.title('En Popüler 7 İsim')

plt.show()

#EN POPÜLER 7 ERKEK İSMİ

veriseti_erkek = veriseti[veriseti['Gender'] == 'M']

populer_isimler_erkek = veriseti_erkek.sort_values(by=['Count'], ascending=False)

en_populer_7_isim_erkek = populer_isimler_erkek.head(7)

renkler = plt.cm.Paired(range(7))

plt.bar(en_populer_7_isim_erkek['Name'], en_populer_7_isim_erkek['Count'], color=renkler)
plt.xlabel('İsimler')
plt.ylabel('Sayı')
plt.title('En Popüler 7 Erkek İsmi')

plt.show()

#EN POPÜLER 7 KADIN İSMİ

veriseti_kadin = veriseti[veriseti['Gender'] == 'F']

populer_isimler_kadin = veriseti_kadin.sort_values(by=['Count'], ascending=False)

en_populer_7_isim_kadin = populer_isimler_kadin.head(7)

renkler = plt.cm.Paired(range(7))

plt.bar(en_populer_7_isim_kadin['Name'], en_populer_7_isim_kadin['Count'], color=renkler)
plt.xlabel('İsimler')
plt.ylabel('Sayı')
plt.title('En Popüler 7 Kadın İsmi')

plt.show()

#ORTALAMA İSİM UZUNLUĞU

def isim_uzunlugu(veriseti):
    veriseti_kopya = veriseti.copy()  # Veri setinin bir kopyasını oluştur

    veriseti_kopya['isim_uzunlugu'] = veriseti_kopya['Name'].apply(len)
    
    ortalama_uzunluk_cinsiyet = veriseti_kopya.groupby('Gender')['isim_uzunlugu'].mean()
    
    genel_ortalama_uzunluk = veriseti_kopya['isim_uzunlugu'].mean()
    
    ortalama_uzunluk = pd.concat([ortalama_uzunluk_cinsiyet, pd.Series({'Genel': genel_ortalama_uzunluk})])

    ortalama_uzunluk.plot(kind='bar', color=['green', 'red', 'black'])
    
    plt.xlabel('Kategori')
    plt.ylabel('Ortalama İsim Uzunluğu')
    plt.title('Cinsiyete Göre ve Genel Ortalama İsim Uzunluğu')

    # Grafiği göster
    plt.show()
    
isim_uzunlugu(veriseti)

#İSİM UZUNLUĞUNA GÖRE FREKANS GRAFİĞİ

def karakter_sayisi_frekansi(veriseti):
    isim_uzunlugu_frekans = veriseti['Name'].apply(len).value_counts().sort_index()
    
    isim_uzunlugu_frekans.plot(kind='bar', color='purple')
    
    plt.xlabel('İsim Uzunluğu')
    plt.ylabel('Kişi Sayısı')
    plt.title('İsim Uzunluğuna Göre Frekans Grafiği')
    
    plt.show()

karakter_sayisi_frekansi(veriseti)

""" KAYIP DEĞERLER İÇİN İŞLEMLER """

veriseti.isnull().sum() #Kayıp değer bulunmamakta

""" VERİLERİN ÖN İŞLENME SÜRECİ """


from sklearn.preprocessing import LabelEncoder

X = veriseti.drop(columns=['Gender']) #Gender sütunu bağımlı değişkenimiz olduğu için çıkardık.

label_encoder = LabelEncoder()

#Label Encoder ile 'Gender' sütununda yer alan F ve M karakterleri 0 ve 1 sayısına dönüştürüldü.
y = label_encoder.fit_transform(veriseti['Gender'])

#Label Encoder ile 'Name' sütununda yer alan karakter dizilerini algoritmaların çalışabileceği format olan sayılara dönüştürüldü.
kategorik_kolonlar = ['Name']
for kolon in kategorik_kolonlar:
    X[kolon] = label_encoder.fit_transform(X[kolon])

#X ve y Veri Kümelerine Göz Atalım

X.head(10)

y[:10]

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Feature Scaling
olcekleyici = StandardScaler()
X_train_olceklendirilmis = olcekleyici.fit_transform(X_train)
X_test_olceklendirilmis = olcekleyici.transform(X_test)

print("X_train")
print(X_train[:10])

print("\nX_train_olceklendirilmis")
print(X_train_olceklendirilmis[:10])

""" SINIFLANDIRMA İŞLEMLERİ """

from sklearn.linear_model import LogisticRegression

#Logistic Regression 
siniflandirici_lr = LogisticRegression(random_state=0)
siniflandirici_lr.fit(X_train_olceklendirilmis, y_train)
y_prediction = siniflandirici_lr.predict(X_test_olceklendirilmis)

#model_sonuc(siniflandirici_lr, y_test, y_prediction)

conf_mat = confusion_matrix(y_test, y_prediction)
acc_scr = accuracy_score(y_test, y_prediction)

print("Logistic Regression Doğruluk Oranı: " + "{:.3f}".format(acc_scr))


#Logistic Regression Sonuç Görselleştirme

from sklearn.metrics import roc_auc_score, roc_curve

logistic_auc = roc_auc_score(y_test, siniflandirici_lr.predict_proba(X_test_olceklendirilmis)[:, 1])
logistic_fpr, logistic_tpr, _ = roc_curve(y_test, siniflandirici_lr.predict_proba(X_test_olceklendirilmis)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(logistic_fpr, logistic_tpr, label=f'Logistic Regression (AUC = {logistic_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression')
plt.legend(loc='lower right')
plt.show()


        

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train_olceklendirilmis, y_train)
rf_pred = rf_model.predict(X_test_olceklendirilmis)

conf_mat = confusion_matrix(y_test, y_prediction)
acc_scr = accuracy_score(y_test, rf_pred)

print("Random Forest Doğruluk Oranı: " + "{:.3f}".format(acc_scr))

rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test_olceklendirilmis)[:, 1])
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_model.predict_proba(X_test_olceklendirilmis)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc='lower right')
plt.show()

# Decision Tree
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_olceklendirilmis, y_train)
dt_pred = dt_model.predict(X_test_olceklendirilmis)

conf_mat = confusion_matrix(y_test, y_prediction)
acc_scr = accuracy_score(y_test, dt_pred)

print("Decision Tree Doğruluk Oranı: " + "{:.3f}".format(acc_scr))

dt_auc = roc_auc_score(y_test, dt_model.predict_proba(X_test_olceklendirilmis)[:, 1])
dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_model.predict_proba(X_test_olceklendirilmis)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(dt_fpr, dt_tpr, label=f'Decision Tree (AUC = {dt_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')
plt.legend(loc='lower right')
plt.show()

from sklearn.neighbors import KNeighborsClassifier

#KNN Algoritması

knn_model = KNeighborsClassifier()
knn_model.fit(X_train_olceklendirilmis, y_train)
knn_pred = knn_model.predict(X_test_olceklendirilmis)

conf_mat = confusion_matrix(y_test, y_prediction)
acc_scr = accuracy_score(y_test, knn_pred)

print("KNN Doğruluk Oranı: " + "{:.3f}".format(acc_scr))

knn_auc = roc_auc_score(y_test, knn_model.predict_proba(X_test_olceklendirilmis)[:, 1])
knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_model.predict_proba(X_test_olceklendirilmis)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(knn_fpr, knn_tpr, label=f'KNN (AUC = {knn_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - KNN')
plt.legend(loc='lower right')
plt.show()

from sklearn.naive_bayes import GaussianNB

# Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train_olceklendirilmis, y_train)
nb_pred = nb_model.predict(X_test_olceklendirilmis)

conf_mat = confusion_matrix(y_test, y_prediction)
acc_scr = accuracy_score(y_test, nb_pred)

print("Naive Bayes Doğruluk Oranı: " + "{:.3f}".format(acc_scr))


nb_auc = roc_auc_score(y_test, nb_model.predict_proba(X_test_olceklendirilmis)[:, 1])
nb_fpr, nb_tpr, _ = roc_curve(y_test, nb_model.predict_proba(X_test_olceklendirilmis)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(nb_fpr, nb_tpr, label=f'Naive Bayes (AUC = {nb_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Naive Bayes')
plt.legend(loc='lower right')
plt.show()

from sklearn.svm import SVC

# Support Vector Machine
svm_model = SVC(probability=True)
svm_model.fit(X_train_olceklendirilmis, y_train)
svm_pred = svm_model.predict(X_test_olceklendirilmis)

conf_mat = confusion_matrix(y_test, y_prediction)
acc_scr = accuracy_score(y_test, svm_pred)

print("Support Vector Machine Doğruluk Oranı: " + "{:.3f}".format(acc_scr))


svm_auc = roc_auc_score(y_test, svm_model.predict_proba(X_test_olceklendirilmis)[:, 1])
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_model.predict_proba(X_test_olceklendirilmis)[:, 1])

plt.figure(figsize=(8, 6))
plt.plot(svm_fpr, svm_tpr, label=f'SVM (AUC = {svm_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - SVM')
plt.legend(loc='lower right')
plt.show()
