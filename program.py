import pandas as pd
import pickle
data = pd.read_csv("C:\\Users\\User\\Desktop\\pycharm\\diabetes.csv")
with open("diabetes.pkl", "wb") as file:
    pickle.dump(data, file)
print("Pickle fayl muvaffaqiyatli yaratildi.")
