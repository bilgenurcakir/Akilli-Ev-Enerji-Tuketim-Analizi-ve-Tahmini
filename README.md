# Akilli Ev Enerji Tuketim Analizi ve Tahmini

# projenin amacı
Bir akıllı evin enerji tüketim verilerini, hava durumu ve zaman faktörleriyle anlık enerji kullanımının normal mi yoksa yüksek mi olduğunu tahmin eden bir model geliştirmektir.

# target
enerji_sev, Evin toplam enerji seviyesini veren use [kW] sutunundan feature engeneering ile elde edilmiştir,tüketim değerleri 0(normal) 1(yüksek) olarak ikiye bölünmüştür.

# neler yapıldı ?
 gerekli ktüphaneler eklendi
'python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
