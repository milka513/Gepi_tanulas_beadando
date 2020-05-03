# Gepi_tanulas_beadando
Gépi tanulás projekt beadandó 2020.

## Fájlok elhelyezkedése
Hogyan lehet reprodukálni a fájlokat.

### Heinc Emília munkája:
#### Adatok feldolgozása:
  1. Először a beolvasas2.py-t kell futtatni. Ez kigenerálja a 12 hónapot a summed_data2.csv-be
  2. Utána a beolvasas3.py-t, ez kiszedi a felelseges feature-ket. Ez a summed_data2.csv-ből a summed_data_vegleges.csv-be menti el.
  3. Utána a feature_processing.ipynb-t. Ez lehet leszedve nem fog lefutni egymagában, előtte kézzel betettem a summed_data_vegleges.csv-t az ideiglenes google drive mappába és úgy futattam le google collab-n. Mivel csak egyszer volt szükséges a lefuttatása, ezért ennek az eredményei megtalálhatóak a summed_data_to_train.csv-ben. A későbbiekben csak a **summed_data_to_train.csv**-t használjuk csak, ebben található a feldolgozott összes eredmény. Minden tanítás során ezt használjuk fel.
  4. bónusz: A createTestTrain.py ami szétszedi test-train halmazra úgy, hogy a files/x_test.csv,y_test.csv,x_train.csv,y_train.csv-be menti ki az adatokat, a későbbiekben ezt olvassuk ki.
#### Ensemble model:
  1. Minden az Ensemble modellel kapcsolatban az **EnsembleModels.py**-ban találhatóak meg. Ezt a fájlt futtatva tanítattam be a modelleimet. Fontos, hogy több óra hosszát is elfut 1 modell tanítattása ezért a kapot betanított modellek adatait kimentettem .pkl fájlokba.
  2. A .pkl fájlok eredményeit az **osszesitett.ipynb**-ban jelenítettem meg és adtam rájuk magyarázatot. Saját gépen nem fog futni mert előtte fel kellett mountolni a meghajtót és a gépi tanulások mappában lévő .pkl fájlokra futtattam le a megjelenítést: https://drive.google.com/drive/folders/1XOYR1u01MvVn4v5ag2cK60basoCsKDmU?usp=sharing
### Baseline taníttatása:
  1. A **baseline_model.py**-ban a tanitas() fv-t kell meghívni a különböző baseline modellek előállításához, ezeket a fv. lementi a módszernevével ellátott pkl kiterjesztésű fájlokba.

### Géczi Dániel munkája:
### Dimenzió redukció
A **dimension_reduction.py**-t kell futtatni és az elvégzi a dimenzió redukciót a **summed_data_to_train.csv**-n és lementi az eredményeket a modellek számára.
A redukció SingleValueDecomposition (SVD)-vel és Principle Component Analysis (PCA)-val történik.

  1. A **model_training.py**-ban található a DecisionTree (DT) model és a SupportVectorRegression (SVR) model és a redukált táblán tanított változataik, a fájl futtatásá értelemszerűen sorban elvégzi a modellek tanítását, de fontos tudni, hogy az SVR modellek
     tanítása nagyon hosszú időt vesz igénybe, mivel paraméter opt. is fel van használva GridSearchCV-t alkalmazva.
  
## Diasor:
bemutato.ppt-t fogjuk bemutatni a bemutatás pont alatt.
  

