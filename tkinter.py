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
