import pandas as pd
#Pandas veri işleme ve analiz işlemleri için import ediyoruz
import matplotlib.pyplot as plt
#Verilerin çıktıların görüntülenmek için kullanılır
import seaborn as sns
#Verilerin çıktıların görüntülenmek için kullanılır
from sklearn.model_selection import train_test_split
#Tahmin işlemleri için genel olarak kullanılan kütüphane
from sklearn.neighbors import KNeighborsClassifier
#Knn algoritması ile tahmin için bu classı çağırıyoruz
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
diabetes = pd.read_csv('diabetes.csv')
#Diyabet csv dosyamızı pandas kütüphanesi ile okutuyoruz.
plt.figure(figsize=(12,10))
#Görsel için boyut ayarlıyoruz.
p=sns.heatmap(diabetes.corr(), annot=True,cmap ='RdYlGn')
#Verilerin ısı haritası üzerinde birbiriyle ilişkisi
plt.show()
#Göstermek için çağırılan fonksiyon.
diabetes.info(verbose=True)
#veri türleri, sütunlar, boş değer sayıları, bellek kullanımı vb. hakkında bilgi verir
print(diabetes.describe())
#Datamız ile ilgili ortalama toplamları gibi sonuçları verir
print(diabetes.shape)
#Verilerin kaç adet satır sütun onu gösterir
print(diabetes.groupby('Outcome').size())
#Outcomede kaç kişi diyabet hastası çıkmış kaç kişi hasta değil ona bakıyoruz. 268 hasta 500 hasta olmayan
p=diabetes.Outcome.value_counts().plot(kind="bar")
#Görsel olarak gösteriyoruz.
plt.show()
#Göstermek için kullanılan fonksiyon.
X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'], diabetes['Outcome'],
stratify=diabetes['Outcome'], random_state=66)
#Makine öğrenmesi öncesi verileri gruplandırma yapıyoruz hangisinin eğitimde hangisi test için kullanılacak
#%66 veriyi eğitim için ayırıyoruz bunu random şekilde seçiyoruz.
training_accuracy = []
#Eğitilecek setin başarısını tanımlıyoruz görselde göstermek için.
test_accuracy = []
#Test edilecek setin başarısını tanımlıyoruz görselde göstermek için.
knneighbors_settings = range(1, 11)
#Knn algoritmasında ki n sayısın da farklı sayıları deneyerek en iyi olanı bulmak için
#for döngüsü içerisine alıp 1-11 arasında n sayılardaki değerleri bulucaz.
for n_neighbors in knneighbors_settings:
#For döngüsü yapılacak yerin içersinde kaç defa tanımlancak onların tanımlanmasını yapıyoruz.
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
#1-11 kadar sayıların knn içerisinde tanımlıyoruz.
    knn.fit(X_train, y_train)
#Fit ile öğrenmek için ayrılan setimize bunu öğretiyoruz.
    training_accuracy.append(knn.score(X_train, y_train))
#Öğrenme için ayırdığımız 2 seti burda ne kadar doğru olduğunu öğrenmek için  ekliyoruz.
    test_accuracy.append(knn.score(X_test, y_test))
#Test için ayırdığımız 2 seti burda ne kadar doğru olduğunu öğrenmek için  ekliyoruz.
plt.title("Başarı Analizi")
#Tablo başlık adı
plt.plot(knneighbors_settings, training_accuracy, label="Öğrenim Seti")
#Görsel olarak göstermek adına x , y kordinatları öğrenme setiyle eşitliyoruz.
plt.plot(knneighbors_settings, test_accuracy, label="Test Seti")
#Görsel olarak göstermek adına x , y kordinatları test setiyle eşitliyoruz.
plt.ylabel("Doğruluk yüzdemiz")
#y düzlemin ismini yazıyoruz.
plt.xlabel("Komşu sayımız")
#x düzlemin ismini yazıyoruz.
plt.legend()
#Grafik elemanları adlandırdığımız etiketleri grafikte gösteriyoruz.
plt.show()
#Grafiği ortaya çıkarıyoruz
knn = KNeighborsClassifier(n_neighbors=9)
#Knn için en uygun olan 9 u seçiyoruz. Sebebi ise komşu sayısının düşük olması bizim modelimiz
#için karmaşıklaştırıyor.Daha fazla komşu seçmek eğitim doğruluğunu düşürmektedir.
#Bu yüzden komşu sayısı fazla olup doğruluk yüksek olan en uygun yer 9 u seçiyoruz.
knn.fit(X_train, y_train)
#En uygun yeri bulduğumuzdan yeni Komşu sayısı 9 seçerek modelimiz buna göre eğitiyirouz.
print('Eğitim setinde K-NN sınıflandırıcısın doğruluk oranı {:.2f}'.format(knn.score(X_train, y_train)))
#Eğitim setinin doğruluk oranını gösteriyoruz.
print('Test setinde K-NN sınıflandırıcısın doğruluk oranı: {:.2f}'.format(knn.score(X_test, y_test)))
#Test setinin doğruluk oranını gösteriyoruz.

#Performans için karışıklık matrisini çağırıyoruz.
y_pred = knn.predict(X_test)
#X testimize göre bir tane tahmini tanımlıyorusz.
confusion_matrix(y_test,y_pred)
#Karışıklık matirisinde tanımladığımızı tahmini y testiyle karşılıklı matris oluşturuyoruz.
print(pd.crosstab(y_test, y_pred, rownames=['Dogruluk'], colnames=['Tahmin'], margins=True))
#Oluşturduğumuz matirisin sütun satır isimleri yazılıp çıktı alınıyor.
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
#Test için ayrılan kısımla tahmini karışıklık matris içine alıyoruz.
p = sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
#Görsel için ayarları yapılıyor.
plt.title('Karışıklık Matrisi', y=1.1)
#Başlık ismi yazıyoruz.
plt.ylabel('Gerçekleşen')
#Y kordinat ismini yazıyoruz.
plt.xlabel('Tahmin')
#Y kordinat ismini yazıyoruz.
plt.show()
#grafiğimizi görselini başlatıyoruz.
print(classification_report(y_test, y_pred))
#F1 Skoru gibi değerleri göstertiyoruz.
