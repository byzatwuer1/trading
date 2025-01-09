import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import joblib
import os
import glob
import pandas_ta as ta  # Teknik indikatörler için gerekli kütüphane

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

# Veri dizini
data_dir = "data"

def load_and_prepare_data():
    """Tüm sembol verilerini yükle ve birleştir"""
    all_data = []
    
    # data dizinindeki tüm CSV dosyalarını bul
    csv_files = glob.glob(os.path.join(data_dir, "*_data.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir} directory")
    
    print(f"Found {len(csv_files)} CSV files")
    
    for file in csv_files:
        try:
            # Sembol adını dosya adından çıkar
            symbol = os.path.basename(file).replace("_data.csv", "")
            
            # CSV dosyasını oku
            df = pd.read_csv(file)
            
            # Sembol sütunu ekle
            df['symbol'] = symbol
            
            # Teknik indikatörleri hesapla ve ekle
            df['RSI'] = ta.rsi(df['close'], length=14)
            macd = ta.macd(df['close'])
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_SIGNAL'] = macd['MACDs_12_26_9']
            bb = ta.bbands(df['close'], length=20, std=2)
            df['BB_UPPER'], df['BB_MIDDLE'], df['BB_LOWER'] = bb['BBU_20_2.0'], bb['BBM_20_2.0'], bb['BBL_20_2.0']
            stochrsi = ta.stochrsi(df['close'], length=14)
            df['StochRSI_K'] = stochrsi['STOCHRSIk_14_14_3_3']
            df['StochRSI_D'] = stochrsi['STOCHRSId_14_14_3_3']
            df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=14)
            df['WILLIAMS_R'] = ta.willr(df['high'], df['low'], df['close'], length=14)
            df['ADX'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
            
            all_data.append(df)
            print(f"Loaded data for {symbol}")
            
        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("No data could be loaded from CSV files")
    
    # Tüm verileri birleştir
    combined_data = pd.concat(all_data, ignore_index=True)
    
    # Tarihe göre sırala
    combined_data['timestamp'] = pd.to_datetime(combined_data['timestamp'])
    combined_data = combined_data.sort_values('timestamp')
    
    # Zaman dilimi özellikleri ekleyin
    combined_data['hour'] = combined_data['timestamp'].dt.hour
    combined_data['day_of_week'] = combined_data['timestamp'].dt.dayofweek
    
    return combined_data

def build_lstm_model(input_shape):
    """LSTM modelini oluştur"""
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

try:
    # Verileri yükle
    print("Loading data...")
    data = load_and_prepare_data()
    
    # Özellikleri hazırla
    print("Preparing features...")
    features = data[['open', 'high', 'low', 'close', 'volume', 'RSI', 'MACD', 'MACD_SIGNAL', 'BB_UPPER', 'BB_MIDDLE', 'BB_LOWER', 'StochRSI_K', 'StochRSI_D', 'ATR', 'WILLIAMS_R', 'ADX', 'hour', 'day_of_week']]
    
    # Etiketleri oluştur (bir sonraki kapanış fiyatı yükselirse 1, düşerse 0)
    labels = (data.groupby('symbol')['close'].shift(-1) > data['close']).astype(int)
    labels = labels[:-1]  # Son satırı kaldır çünkü next_close değeri yok
    features = features[:-1]  # Etiketlerle eşleşmesi için son satırı kaldır
    
    # Veriyi ölçekle
    print("Scaling features...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(features)
    
    # Eğitim ve test setlerine ayır
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(scaled_features, labels, test_size=0.2, random_state=42)
    
    # LSTM modelini oluşturma
    input_shape = (X_train.shape[1], 1)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    model = build_lstm_model(input_shape)
    
    # Modeli eğitme
    print("Training model...")
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))
    
    # Model performansını değerlendir
    train_score = model.evaluate(X_train, y_train)[1]
    test_score = model.evaluate(X_test, y_test)[1]
    print(f"Training accuracy: {train_score:.4f}")
    print(f"Testing accuracy: {test_score:.4f}")
    
    # Modeli ve scaler'ı kaydet
    print("Saving model and scaler...")
    model.save('models/lstm_model.h5')
    joblib.dump(scaler, 'models/scaler.pkl')
    
    print("Model and scaler successfully saved.")

except Exception as e:
    print(f"Error during training: {str(e)}")
    raise  # Re-raise the exception for debugging

piyasa dinamiklerini okuyup anlayabilmeli  binance trader tahminleri güvenilir olmalı 
