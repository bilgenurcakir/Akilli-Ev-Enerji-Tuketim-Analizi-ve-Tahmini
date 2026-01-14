# Akilli Ev Enerji Tuketim Analizi ve Tahmini


# alınan BTK akademi Makine öğrenmesi sertifikası




# projenin amacı
Bir akıllı evin enerji tüketim verilerini, hava durumu ve zaman faktörleriyle anlık enerji kullanımının normal mi yoksa yüksek mi olduğunu tahmin eden bir model geliştirmektir.

# target
enerji_sev, Evin toplam enerji seviyesini veren use [kW] sutunundan feature engeneering ile elde edilmiştir,tüketim değerleri 0(normal) 1(yüksek) olarak ikiye bölünmüştür.

# neler yapıldı ?

 gerekli ktüphaneler eklendi.
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
```
dataset alındı ve incelendi
```python
df=pd.read_csv("Smart_Home_dataset.csv")
df.head()
df.describe()
df.info()
```
dataset içerisindeki kategorik kolonlar incelendi ve numerik hale getirildi

-- coloudcover kolonu incelendiğinde sayısal olduğu ancak bir adet "cloudCover" değişkeninin bulunduğu görüldü ve Nan olarak değiştirildi.
daha sonrasında numeric hale çevirildi.
```python
df.cloudCover.unique()
df.cloudCover=df.cloudCover.replace('cloudCover',np.nan)
df.cloudCover.unique()
df.cloudCover=pd.to_numeric(df.cloudCover)
```
-- time kolonu incelendiğinde 10 basamakli saniye formatında verildiği görüldü ve datetime ile tarih formatına getirilerek model için önemli olacak( enerji tüketimi aylara, haftanın günlerine ve saate göre değişiklik gösterebilir bu yüzden) month, day_of_week ve hour feature'ları alındı ve time değikeni silindi .
```python
df.time.unique # datetime ile yeniden formatlanmalı
df['time'] = pd.to_numeric(df['time'], errors='coerce')
df['time'] = pd.to_datetime(df['time'],unit='s', errors='coerce')
df['hour'] = df['time'].dt.hour
df['day_of_week'] = df['time'].dt.dayofweek 
df['month'] = df['time'].dt.month
df.drop( columns=["time"], inplace=True)
```
-- summary kolonu incelendiğinde sayısal değer içermediği görüldü ve label encoding yapıldı.(label encoding pivot tablosu gösteriminden sonra yapıldı böylece pivot tablosunda hava durumu yerine sayısal bir değer yazmadı, inceleme kolaylaştırıldı)
```python
df.summary.unique()
df.summary.unique #label encoding yapılacak
le=LabelEncoder()
df["summary"]=le.fit_transform(df["summary"])
df.summary.unique()
```

--icon kolonu incelendiğinde gereksiz görüldü ve dataframeden çıkarıldı.(summary ile aynı değerleri içeren, uygulamalardaki hava durumu ikonunu gösteren kolon)

```python
df.icon.unique() # ikon atılabilir gereksiz
df.drop(columns=['icon'], inplace=True)
```

bu işlemlerden sonra Nan değer kontrolü yapıldı, nan değerlerin az olması sebebiyle dataframeden çıkarıldı.

```python
df.isnull().sum()
df=df.dropna()
```

pivot tablolama- zaman ve hava durumu verilerini doğrudan kullanmak yerine use[kW] üzerindeki ağırlığını yanıtan pivot tabloları oluşturuldu, dataframe eklendi


```python
pivot=df.pivot_table(index="hour", values="use [kW]", aggfunc="mean").reset_index() # saat bazlı ortalaa tüketim pivot tablosu

plt.figure(figsize=(8,8))
sns.barplot(x='use [kW]', y='hour', data=pivot.sort_values(by='use [kW]', ascending=False))
plt.title("Saatlere Göre Enerji Tüketimi")
plt.xlabel("Ortalama Tüketim (kW)")
plt.ylabel("saat")
plt.tight_layout()
plt.show()

pivot2=df.pivot_table(index="summary", values="use [kW]", aggfunc="mean").reset_index() # hava durumu bazlı ortalama tüketim pivot tablosu

plt.figure(figsize=(8,8))
sns.barplot(x='use [kW]', y='summary', data=pivot2.sort_values(by='use [kW]', ascending=False))
plt.title("Hava Durumuna Göre Ortalama Enerji Tüketimi")
plt.xlabel("Ortalama Tüketim (kW)")
plt.ylabel("Hava Durumu")
plt.tight_layout()
plt.show()

pivot.columns=["hour","saatlik_tuketim"]
pivot2.columns=["summary", "hava_tuketim"]

df = df.merge(pivot, on='hour', how='left')
df = df.merge(pivot2, on='summary', how='left')
df.info()
```

target, use[kW] feature'ı üzerinden ortalama alınarak ortalama uzerinde kalanlara yüksek(1), altında kalanlara ise normal(0) değerini alan feature oluşturuldu

```python
ortalama=df['use [kW]'].median()
df["enerji_sev"]=(df["use [kW]"]> ortalama).astype(int) # ortalamadan az ise 0 cok ise 1
```
modele verilecek girdiler ve target seçildi, train ve test olarak ayrıldı.

```python
alma = ['use [kW]', 'gen [kW]', 'House overall [kW]', 
        'Dishwasher [kW]', 'Furnace 1 [kW]', 'Furnace 2 [kW]', 
        'Home office [kW]', 'Fridge [kW]', 'Wine cellar [kW]', 
        'Garage door [kW]', 'Kitchen 12 [kW]', 'Kitchen 14 [kW]', 
        'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]', 
        'Microwave [kW]', 'Living room [kW]', 'Solar [kW]',
        'apparentTemperature', 'dewPoint', 'precipProbability', 
        'summary',"hour", 'enerji_sev']
# sonunca [kW] içerenler modelde kullanılamaz target'ı içerir, 
#apparentTemperature yerine tempature kullandık gereksiz, 
#dewPoint:çiğ düşme noktası humidity ve tempute'ın matematiksel hesabı gereksiz,
#precipProbability ve precipIntensity birbirine çok benzer,
#hour ve summary yerine pivot tablosuyla oluşturduğumuz saatlik_tüketim, hava_tüketim kullanılacak

X=df.drop(columns=alma)
y=df["enerji_sev"]

X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.2, random_state=42)
```
RandomForest modeli kullanıldı, modelin hangi özelliklere daha çok dikkat ettiği grafikleştirildi.

```python
model=RandomForestClassifier()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
acc=accuracy_score(y_test,y_pred)
classificationtable=classification_report(y_test,y_pred)
print("acc degeri:",acc)
print("sonuc:",classificationtable)

# Modelden önem derecelerini al
importances = model.feature_importances_
featurenames = X.columns
onem = pd.DataFrame({'Özellik': featurenames, 'Önem': importances}).sort_values(by='Önem', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Önem', y='Özellik', data=onem)
plt.title('Random Forest modelin Karar Verirken En Çok Önem Verdiği Özellikler')
plt.xlabel('Önem derecesi')
plt.ylabel('feature')
plt.show()
```

KNN modeli çağrıldı, öncesinde daha iyi bir sonuç için StandardScaler ile X_train ve X_test ölçeklendirildi

```python
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
knn=KNeighborsClassifier()
knn.fit(X_train,y_train)
y_knn_pred=knn.predict(X_test)

acc2=accuracy_score(y_test,y_knn_pred)
classificationtable2=classification_report(y_test,y_knn_pred)
print("acc degeri:",acc2)
print("sonuc:",classificationtable2)

```


model çıktıları:

1- randomforest
acc degeri: 0.8342380248285717
sonuc:               precision    recall  f1-score   support

           0       0.83      0.84      0.84     50597
           1       0.83      0.83      0.83     50174

    accuracy                           0.83    100771
   macro avg       0.83      0.83      0.83    100771
weighted avg       0.83      0.83      0.83    100771

2- KNN
acc degeri: 0.814341427593256
sonuc:               precision    recall  f1-score   support

           0       0.81      0.82      0.82     50597
           1       0.81      0.81      0.81     50174

    accuracy                           0.81    100771
   macro avg       0.81      0.81      0.81    100771
weighted avg       0.81      0.81      0.81    100771



grafikler:



