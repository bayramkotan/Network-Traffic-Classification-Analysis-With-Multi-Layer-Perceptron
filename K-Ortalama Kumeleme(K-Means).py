import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
import psutil
import os


print("K-Means")
# Train ve Test Verisetlerini Oku
train = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Derin Öğrenme ile Ağ İzleme Sistemi ve Karşılaştırmalı Analizi/Train_Verileri.csv', header=None)
test = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Derin Öğrenme ile Ağ İzleme Sistemi ve Karşılaştırmalı Analizi/Test_Verileri.csv', header=None)


# Baslıklar.csv dosyasından kolon adlarını oku ve train ve test veri setlerinde kolon başlığı olarak ata
columns = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Derin Öğrenme ile Ağ İzleme Sistemi ve Karşılaştırmalı Analizi/Basliklar.csv', header=None)
columns.columns = ['name', 'type']
train.columns = columns['name']
test.columns = columns['name']


# Servis tiplerini oku 
serviceType = pd.read_csv('/content/drive/My Drive/Colab Notebooks/Derin Öğrenme ile Ağ İzleme Sistemi ve Karşılaştırmalı Analizi/Servis_Tipleri.csv', header=None)
serviceType.columns = ['Name', 'Type']
serviceMap={}



# (Sql join işlemine benzer) Mapping işlemi yap.
for i in range(len(serviceType)):
    serviceMap[serviceType['Name'][i]] = serviceType['Type'][i]
print("Mapping İşlemi Yapıldı")



# train ve test verilerinde label kolonu oluştur ve mapping işlemini bu kolona bağla yani sonuç kolonu (etiketleme kolonu)
train['label'] = train['service'].map(serviceMap)
test['label'] = test['service'].map(serviceMap)



#Servislerin sayısını bul
classesCount = len(train['label'].drop_duplicates())
classesName = train['label'].drop_duplicates().values.tolist()
print('Sınıf sayısı:' +  str(classesCount) )
print('Sınıf İsimleri :')
print(classesName)



# LabelEncoder kullanarak nominal verileri nümerik değerlere çevir
for col in ['protocol_type', 'flag', 'attack_type', 'label']:
    le = LabelEncoder()
    le.fit(train[col])
    train[col] = le.transform(train[col])
    le1 = LabelEncoder()
    le1.fit(test[col])
    test[col] = le1.transform(test[col])



# Test ve Train verilerindeki label kolonu etiket kolonumuz olduğundan farklı değişkenlerde tutalım
trainLabel = train['label']
testLabel = test['label']



# service ve label kolonları sonuç kolonları olduğundan bunları verisetlerinden atalım
train.drop(['service', 'label'], axis=1, inplace=True)
test.drop(['service', 'label'], axis=1, inplace=True)



# Stadart Scaler ile verileri 0-1 aralığına ölçekliyelim
scaler = MinMaxScaler()  
train = scaler.fit_transform(train)
test = scaler.fit_transform(test)      
total = np.concatenate([train, test] )



# PCA - Temel Bileşenler Analizi
pca = PCA(n_components=41, random_state=100)
pca.fit(total)
train = pca.transform(train)
test = pca.transform(test)
print("Kullanılan Özellik Sayısı : %d" % train.shape[1])


# K katlamalı çapraz doğrulama
startTime = time.clock()
KM1 = KMeans(n_clusters = classesCount, init = 'random', random_state=100)
KM1.fit(train, trainLabel)
score = abs(cross_val_score(KM1, train, trainLabel, cv=5))/1000
endTime = time.clock()
print("5-Katlamalı çapraz doğrulamanın gerçekleştirim zamanı : %f" % (endTime - startTime))
print("5-Katlamalı çapraz doğrulama oranları : ")
print(score)
print("5-Katlamalı çapraz doğrulamanın ortalaması: %% %.2f" % abs(score.mean()/1000))



# Modelin eğitilmesi
startTime = time.clock()
KM2 = KMeans(n_clusters = classesCount, init = 'k-means++' ) 
KM2.fit(train, trainLabel)
endTime = time.clock()
print("Modelin eğitiminin gerçekleştirim süresi : %f" % (endTime - startTime))


# Modelin test edilmesi
startTime = time.clock()
pred = KM2.predict(test)
endTime = time.clock()
cpuUsage = psutil.cpu_percent()
pid = os.getpid()
py = psutil.Process(pid)
memoryUse = py.memory_info()[0] / 2. ** 30
print("Modelin testinin gerçekleştirim süresi : %f" % (endTime - startTime))
print("Kullanılan hafıza : %f GB  , Kullanılan işlemci : %f" % (memoryUse, cpuUsage))




#Karışıklık Matrisi
con_matrix = confusion_matrix(y_pred=pred, y_true=testLabel) 
print("Karışıklık Matrisi : ")
print(con_matrix)


# Accuracy ve detection rate değerşerini hesapla
acc = accuracy_score(y_pred=pred, y_true=testLabel)
print("Test verisinin ACC değeri : %f" % acc)

sumDr = 0
for i in range(con_matrix.shape[0]):
    det_rate = 0
    for j in range(con_matrix.shape[1]):
        if i != j :
            det_rate += con_matrix[i][j]
    if con_matrix[i][i] != 0 or (det_rate + con_matrix[i][i])  != 0:
        det_rate =100* con_matrix[i][i]/(det_rate + con_matrix[i][i])
        sumDr += det_rate


DR = sumDr/classesCount
print("Test verisinin Detection Rate değeri % " + str(DR))