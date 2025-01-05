import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Veriyi yükleme
def load_data():
    # Yerel dosya yolunu belirtin
    df = pd.read_csv('cancer_reg.csv')  # Bu dosya yolunu doğru şekilde belirtin
    return df

# Gürültü giderme fonksiyonu
def smooth_data(data, window_size=3):
    return data.rolling(window=window_size, min_periods=1).mean()

def train_model_2():
    # Veri setini yükle
    df = load_data()

    # Gürültü giderme işlemi
    columns_to_smooth = [
        'avganncount', 'avgdeathsperyear', 'incidencerate', 'medincome',
        'popest2015', 'povertypercent', 'studypercap', 'medianage',
        'pctprivatecoveragealone', 'pctempprivcoverage', 'pctpubliccoverage',
        'pctpubliccoveragealone', 'pctwhite', 'pctblack', 'pctasian',
        'pctotherrace', 'pctmarriedhouseholds', 'birthrate'
    ]
    
    for col in columns_to_smooth:
        df[f"{col}_smoothed"] = smooth_data(df[col])

    # Bağımsız ve bağımlı değişkenleri belirleme
    target = 'target_deathrate'
    features = [f"{col}_smoothed" for col in columns_to_smooth]

    # Eksik verileri doldurma
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].mean(), inplace=True)

    # Normalizasyon (0-1 ölçeğine dönüştürme)
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[features + [target]])
    df_scaled = pd.DataFrame(df_scaled, columns=features + [target])

    # Veri setini ayırma
    X = df_scaled[features]
    y = df_scaled[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Regresyon modeli oluşturma
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Model tahminleri
    y_pred = model.predict(X_test)

    # Performans değerlendirme
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Sonuçları çizme
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label="Gerçek Değerler", marker='o')
    plt.plot(y_pred, label="Tahminler", linestyle='dashed', marker='x')
    plt.xlabel("Örnekler")
    plt.ylabel("Normalized Target Death Rate")
    plt.legend()
    plt.title("Linear Regression Tahminleri ve Gerçek Değerler")
    plt.grid()

    # Grafik kaydetme
    plot_path = 'static/regression_results_model2.png'  # Grafik kaydedilecek dizin
    plt.savefig(plot_path)
    plt.close()

    return mse, plot_path
