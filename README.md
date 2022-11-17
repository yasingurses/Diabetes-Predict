


# HASTA TAHLİL VERİ SETİ KULLANARAK ŞEKER HASTALIĞINI TAHMİN ETME










 # 1.	PROJEDE KARŞILAŞILAN SORUNLAR, ÇÖZÜMLER VE ÇÖZÜM SÜRESİ

KARŞILAŞILAN SORUNLAR	ÇÖZÜMLER	ÇÖZÜM SÜREMİZ
IDE olarak PyCharm kullanırken kütüphane yükleme sıkıntısı çektik.	Stackoverflow, Medium gibi çeşitli yazılı internet kaynaklarından yararlandık.	2 Saat
Python kütüphanesindeki bir bug yüzünden programın doğruluğundan şüphe duyulması.	Karşılaşılan sorunun araştırılması ve herhangi bir sorun teşkil etmediği kütüphaneden kaynaklanan bir sorun olduğunun tespit edilmesi	2 Saat
Projede hangi yapay zeka tekniğinin uygulanmasının seçimi.	En çok doğruluk payı çıkaran sonucu kullandık	5 Gün
Python arayüz (Tkinter) kullanımı ve Data setin entegre edilmesi.	Tkinter kullanımının öğrenilmesi üzerine çalışmalar gerçekleştirdik. 	3 Gün

# 2.	PROJEDE KULLANILAN TEKNİKLER VE YÖNTEMLER
KNN, makine öğrenmesi kullanılmıştır. Veri setindeki veriler bu sayede kullanılan tekniğimizde hata payı azaltılması hedeflenmiştir. 


# 3.	PROJE TANIMI
Diyabet verisi üzerine makine öğrenmesi tekniğini kullanarak kişilerin KNN algoritmasına göre diyabetli ya da diyabetsiz olduklarını tahmin etme programı geliştirilmiştir. Bu kapsamda geliştirilen bir arayüz ile kullanımı kolaylaştırılması hedeflenmiştir.

# 4.	PROJE AMACI
Diyabet, vücudumuzda pankreas adlı salgı bezinin yeterli miktarda insülin hormonu üretememesi ya da ürettiği insülin hormonunun etkili bir şekilde kullanılamaması sonucunda gelişir. Tüketilen besinlerden kana geçen şeker hücreler tarafından kullanılamadığı için kan şekeri yükselir. Kan şekeri kontrol altına alınmadığı takdirde zaman içerisinde diyabet hastalığı körlüğe, kalp ve damar hastalıklarına, inmeye (felç), böbrek yetmezliğine ve sinir sisteminde hasara yol açabilir. Aynı şekilde gebelik döneminde de kontrol altına alınamayan diyabet anne ve bebek sağlığı açısında sağlık sorunlarına neden olabilmektedir.
Projemizin amacı diyabet hastası olma şüphesi duyan ve ön test yapmak isteyen kişilere yardımcı olmak ve olası gelişecek sorunları engellemesi amaçlanmıştır.

# 5.	PROJE AKIŞ DİYAGRAMI
 
# 6.	PROJEDE KULLANILAN PROGRAM VE KÜTÜPHANELER
Programlama dili olarak python tercih edilmiştir. Python tercih edilme sebebi yapay zeka projeleri için birçok kütüphanenin entegre bir şekilde çalışıyor olması ve python dilinin kullanımının sade ve anlaşılır olmasından kaynaklanmaktadır. Kullandığımız python kütüphaneleri; 
pandas, sklearn, matplotlib, seaborn ve tkinter.

# 7.	PROJE KODLARIMIZ (AÇIKLAMA SATIRLARI İLE BİRLİKTE)
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



TKİNTER (GUI)
import tkinter
from tkinter import *
from tkinter import messagebox
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def gonder():
    glukoz1 = degisken1_tipi_opsiyon.get()
    derikalin1 = degisken2_tipi_opsiyon.get()
    dogum1 = degisken3_tipi_opsiyon.get()
    kan1 = degisken4_tipi_opsiyon.get()
    ins1 = degisken5_tipi_opsiyon.get()
    vct1 = degisken6_tipi_opsiyon.get()
    yas1 = degisken7_tipi_opsiyon.get()
    aile1 = degisken8_tipi_opsiyon.get()

    diabetes = pd.read_csv('diabetes.csv')

    X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'],
                                                        diabetes['Outcome'], stratify=diabetes['Outcome'],
                                                        random_state=66)

    training_accuracy = []
    test_accuracy = []
    neighbors_settings = range(1, 11)
    for n_neighbors in neighbors_settings:
        knn = KNeighborsClassifier(n_neighbors=n_neighbors)
        knn.fit(X_train, y_train)
        training_accuracy.append(knn.score(X_train, y_train))
        test_accuracy.append(knn.score(X_test, y_test))
    knn = KNeighborsClassifier(n_neighbors=9)
    knn.fit(X_train, y_train)
    Xn = [[dogum1, glukoz1, kan1, derikalin1, ins1, vct1, aile1, yas1]]
    out = knn.predict(Xn)
    if out == 0:
        print("Diyabet hastası değilsiniz ")
        tkinter.messagebox.showinfo(title="Sonuc", message="Diyabet hastası değilsiniz")

    else:
        tkinter.messagebox.showinfo(title="Sonuc", message="Üzgünüz diyabet hastasınız")
        print("Üzgünüz diyabet hastasınız ")
glkz= [i for i in range(1001)]
dgm= [i for i in range(11)]
derikalin= [i for i in range(101)]
derikalin= [i for i in range(101)]
kanbasinc= [i for i in range(151)]
insulinn= [i for i in range(1001)]
endeks= [i for i in range(672)]
yas = [i for i in range(121)]
kalitim = [i for i in range(2501)]
master = Tk()
master.title('Diyabet Tahmin Programı')
canvas = Canvas(master, height=650, width=950)
canvas.pack()
frame_ust = Frame(master)
frame_ust.place(relx=0.1, rely=0.1, relheight=0.1, relwidth=0.75)
frame_ust2 = Frame(master)
frame_ust2.place(relx=0.1, rely=0.3, relheight=0.1, relwidth=0.75)
frame_ust3 = Frame(master)
frame_ust3.place(relx=0.1, rely=0.5, relheight=0.1, relwidth=0.75)
frame_ust4 = Frame(master)
frame_ust4.place(relx=0.1, rely=0.7, relheight=0.1, relwidth=0.75)
frame_ust5 = Frame(master)
frame_ust5.place(relx=0.1, rely=0.9, relheight=0.1, relwidth=0.75)
degisken1_etiket = Label(frame_ust, text="Glukoz Değeri: ", font="SFpro 15 bold")
degisken1_etiket.pack(padx=10, pady=10, side=LEFT)
degisken1_tipi_opsiyon = StringVar(frame_ust)
degisken1_tipi_opsiyon.set("\t")
degisken1_acilir_menu = OptionMenu(
    frame_ust,
    degisken1_tipi_opsiyon,
    *glkz
)
degisken1_acilir_menu.pack(padx=10, pady=10, side=LEFT)
#################################
degisken2_tipi_opsiyon = StringVar(frame_ust)
degisken2_tipi_opsiyon.set("\t")
degisken2_acilir_menu = OptionMenu(
    frame_ust,
    degisken2_tipi_opsiyon,
    *derikalin
)
degisken2_acilir_menu.pack(padx=10, pady=10, side=RIGHT)
degisken2_etiket = Label(frame_ust, text="Deri kalınlığı: ", font="SFpro 15 bold")
degisken2_etiket.pack(padx=10, pady=10, side=RIGHT)
###############################
degisken3_etiket = Label(frame_ust2, text="Kaç Doğum Yaptınız: ", font="SFpro 15 bold")
degisken3_etiket.pack(padx=10, pady=10, side=LEFT)
degisken3_tipi_opsiyon = StringVar(frame_ust2)
degisken3_tipi_opsiyon.set("\t")
degisken3_acilir_menu = OptionMenu(
    frame_ust2,
    degisken3_tipi_opsiyon,
    *dgm
)
degisken3_acilir_menu.pack(padx=10, pady=10, side=LEFT)

###############################
degisken4_tipi_opsiyon = StringVar(frame_ust2)
degisken4_tipi_opsiyon.set("\t")

degisken4_acilir_menu = OptionMenu(
    frame_ust2,
    degisken4_tipi_opsiyon,
    *kanbasinc
)
degisken4_acilir_menu.pack(padx=10, pady=10, side=RIGHT)

degisken4_etiket = Label(frame_ust2, text="Kan Basıncı: ", font="SFpro 15 bold")
degisken4_etiket.pack(padx=10, pady=10, side=RIGHT)

###############################
degisken5_etiket = Label(frame_ust3, text="İnsulin Değeri: ", font="SFpro 15 bold")
degisken5_etiket.pack(padx=10, pady=10, side=LEFT)
degisken5_tipi_opsiyon = StringVar(frame_ust3)
degisken5_tipi_opsiyon.set("\t")

degisken5_acilir_menu = OptionMenu(
    frame_ust3,
    degisken5_tipi_opsiyon,
    *insulinn
)
degisken5_acilir_menu.pack(padx=10, pady=10, side=LEFT)


###############################
degisken6_tipi_opsiyon = StringVar(frame_ust2)
degisken6_tipi_opsiyon.set("\t")

degisken6_acilir_menu = OptionMenu(
    frame_ust3,
    degisken6_tipi_opsiyon,
    *endeks
)
degisken6_acilir_menu.pack(padx=10, pady=10, side=RIGHT)

degisken6_etiket = Label(frame_ust3, text="Vücut Kitle Endeksi: ", font="SFpro 15 bold")
degisken6_etiket.pack(padx=10, pady=10, side=RIGHT)

###############################
degisken7_etiket = Label(frame_ust4, text="Yaşınız: ", font="SFpro 15 bold")
degisken7_etiket.pack(padx=10, pady=10, side=LEFT)
degisken7_tipi_opsiyon = StringVar(frame_ust4)
degisken7_tipi_opsiyon.set("\t")

degisken7_acilir_menu = OptionMenu(
    frame_ust4,
    degisken7_tipi_opsiyon,
    *yas
)
degisken7_acilir_menu.pack(padx=10, pady=10, side=LEFT)


###############################

degisken8_tipi_opsiyon = StringVar(frame_ust4)
degisken8_tipi_opsiyon.set("\t")

degisken8_acilir_menu = OptionMenu(
    frame_ust4,
    degisken8_tipi_opsiyon,
    *kalitim
)
degisken8_acilir_menu.pack(padx=10, pady=10, side=RIGHT)

degisken8_etiket = Label(frame_ust4, text="Aile Diyabet Kalıtım Değeri: ", font="SFpro 15 bold")
degisken8_etiket.pack(padx=10, pady=10, side=RIGHT)

###############################BUTON###############################

gonder_butonu = Button(frame_ust5, text="Test Et", command=gonder)
gonder_butonu.pack(anchor=CENTER)

degisken9_etiket = Label(frame_ust5, text="Rojhat Birel & Lütfi Gürses & Cem Tatlı ", font="SFpro 14 bold italic ", fg= "green")
degisken9_etiket.pack(padx=10, pady=10, side=LEFT)

master.mainloop()

# 8.	KULLANILAN VERİ SETİ TANITIMI

•	Bu veri seti Hindistan merkezli bir hastaneden alınmıştır
•	veri seti kaggle.com üzerinden indirilmiştir daha sonra kullanılmak üzere kontrolü yapılıp işlenmiştir.
•	Veri setindeki veri kümeleri, birkaç tıbbi(bağımsız) değişkenden ve bir hedef (bağımlı) değişken olan sonuçtan oluşur. Bağımsız değişkenler, hastanın sahip olduğu gebelik sayısı, vücut kitle endeksi (BMI), insülin düzeyi, yaşı, aile kalıtım mirası, Deri kalınlığı ve en son olarak verilerin sahiplerinin diyabet olup olmadığını belirten bir değer mevcuttur.
 

OUTCOME ÇIKTISI
 
# 9.	PERFORMANS DEĞERLENDİRME SÜRECİ
 
 

F1 SKORU

 




BAŞARI ORANI
 

# 10.	AYNI VERİ SETİ İLE YAPILAN FARKLI ÇALIŞMALARLA KIYASLAMA

Birçok farklı model kullanıldığını gördük. Bu modeller;
1.Logistic Regression 
Lojistik regresyon, bir sonucu belirleyen bir veya daha fazla bağımsız değişken bulunan bir veri kümesini analiz etmek için kullanılan istatistiksel bir yöntemdir. Sonuç, ikili bir değişkenle ölçülür (yalnızca iki olası sonuç vardır).
2.Random Forest Classifier
Rastgele ormanlar veya rastgele karar ormanları, sınıflandırma, regresyon ve diğer görevler için, eğitim aşamasında çok sayıda karar ağacı oluşturarak problemin tipine göre sınıf (sınıflandırma) veya sayı (regresyon) tahmini yapan bir toplu öğrenme yöntemidir
3.Support Vector Machine 
Destek vektör makinesi, eğitim verilerindeki herhangi bir noktadan en uzak olan iki sınıf arasında bir karar sınırı bulan vektör uzayı tabanlı makine öğrenme yöntemi olarak tanımlanabilir.

En yüksek accuracy (doğruluk) KNN modelinde çıktığınız gözlemledik. Başka bir husus ise yine çoğu projede bir arayüz entegresi ile bu modeli test etme imkanı yoktu ya da biz araştırmalarımız sonucunda rastlamadık. Bizde bir arayüz tasarlayarak hem kullanım kolaylığı hem de görsellik katarak projemizi bu şekilde geliştirdik.




# 11.	KULLANILAN VERİ SETİ İLE BAŞKA HANGİ ÇALIŞMALAR YAPILABİLİR?

Bu veri seti kişinin belirli medikal verilerini (Kan basıncı, insülin, vücut kitle endeksi vb.) ve diyabet hastası olup olmadığının verisini içermektedir. Bu sebeplerden ötürü yine başka çalışmaları incelediğimizde bu proje kapsamında da yapıldığı gibi eldekiler verileri test ve train edilerek kişinin hasta olup olmadığı tespiti yapılmak üzerine çalışmalar olduğunu görmekteyiz.

KAYNAKÇA

PANDAS
https://www.veribilimiokulu.com/k-en-yakin-komsu-k-nearest-neighbor-siniflandirma-python-ornek-uygulama/ 
https://www.youtube.com/watch?v=xv-1ax50BKM 
MATPLOTLİB
https://matplotlib.org/stable/tutorials/index 
SKLEARN
https://www.youtube.com/watch?v=IlMzkTcIqjA 
TKİNTER
https://www.tutorialspoint.com/python/python_gui_programming.htm
