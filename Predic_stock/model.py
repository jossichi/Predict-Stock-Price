
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from datetime import datetime, timedelta
import re
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from tensorflow import keras
from keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings("ignore")

class LSTM:
    def __init__(self, file_path, sheet_name, target_column='Close', feature_columns=None, test_size=0.2, current_date=None, learning_rate=0.001 , n_iterations=100, random_state=None,scaler=None):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.test_size = test_size
        self.current_date = current_date or datetime.now().strftime('%Y-%m-%d')
        self.df = {}  
        self.analysis_results = {}
        self.next_day = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        self.model = None
        self.merged_df = None
        self.X = None
        self.y = None
        self.model = None
        self.df_spliting = {}
        self.df_training = []
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.random_state = random_state
        self.scaler = scaler  

    def analyze_data(self):
            if os.path.exists(self.file_path):
                print(f"File '{self.file_path}' tồn tại.")

                # Đọc thông tin về các sheets trong file Excel
                excel_sheets = pd.ExcelFile(self.file_path).sheet_names

                # In ra số lượng sheets và tên của từng sheet
                print(f"Số lượng sheets: {len(excel_sheets)}")
                print(f"Danh sách các sheets: {excel_sheets}")
                for sheet_name in excel_sheets:
                    print(f"- {sheet_name}")

                    # Đọc dữ liệu từ mỗi sheet
                    sheet_df = pd.read_excel(self.file_path, sheet_name)
                    
                    
                        
                    if not sheet_df.empty:
                        
                        # Update self.df with processed data
                        self.df[sheet_name] = sheet_df
                        print(f"dataframe: {self.df}")

                print("Duyệt sheet hoàn tất")            

            elif not self.df:
                print("Không có dữ liệu để kiểm tra.")
                return
        
    def split_data(self, n_splits=5, output_file_path='processed_stock_data.xlsx'):
        # Đọc tất cả các sheet từ file Excel vào DataFrame
        xls = pd.ExcelFile(output_file_path)
        all_sheet_names = xls.sheet_names
        
        #Lưu toàn bộ dữ liệu trong sheet vào self.df
        print(f"Dataframe before splitting data: \n{self.df}")
        # Lưu trữ dữ liệu của các sheet có tên '%Y-%m-%d' vào DataFrame
        for sheet_name in all_sheet_names:
            if re.match(r'\d{4}-\d{2}-\d{2}', sheet_name):
                try:
                    df_sheet = pd.read_excel(xls, sheet_name)
                    self.df_spliting[sheet_name] = df_sheet 

                except ValueError:
                    print(f"Sheet '{sheet_name}' không phải là định dạng ngày '%Y-%m-%d'. Skipping...")

        print(f"dataframe splitting \n {self.df_spliting}")

        # Lấy danh sách sheet_name có định dạng '%Y-%m-%d' từ DataFrame
        valid_sheet_names = [name for name in self.df_spliting.keys()]
        print(f"valid_sheet_names {valid_sheet_names}")

        # Chọn 80% làm training data, 20% làm testing data
        valid_sheet_names.sort()
        training_size = int(0.8 * len(valid_sheet_names))
        training_sheet_names = valid_sheet_names[:training_size]
        print(f"training_sheet_names {training_sheet_names}")
        
        training_data = pd.concat([self.df_spliting[name] for name in training_sheet_names])        
        print(f"Training Data:\n{training_data}")
        
        testing_sheet_names = valid_sheet_names[training_size:]
        print(f"testing_sheet_names {testing_sheet_names}")
        
        testing_data = pd.concat([self.df_spliting[name] for name in testing_sheet_names])
        print(f"Testing Data:\n{testing_data}")
        
        # Áp dụng TimeSeriesSplit cho mỗi sheet trong training_data
        tscv = TimeSeriesSplit(n_splits=n_splits)
        X_train_list, X_test_list, y_train_list, y_test_list = [], [], [], []

        for sheet_name in training_sheet_names:
            selected_df = self.df_spliting[sheet_name]

            X_train = selected_df[self.feature_columns]
            y_train = selected_df[self.target_column]

            X_train_list.append(X_train)
            y_train_list.append(y_train)

        # Lấy dữ liệu kiểm tra từ các sheet trong testing_sheet_names
        for sheet_name in testing_sheet_names:
            selected_df = self.df_spliting[sheet_name]

            X_test = selected_df[self.feature_columns]
            y_test = selected_df[self.target_column]

            X_test_list.append(X_test)
            y_test_list.append(y_test)

        print(f"X_train_list: {X_train_list}")
        print(f"X_test_list: {X_test_list}")
        print(f"y_train_list: {y_train_list}")
        print(f"y_test_list: {y_test_list}")
        return X_train_list, X_test_list, y_train_list, y_test_list
    
    def calculate_total_rows(self):
        if not self.df:
            print("No data to calculate.")
            return

        # Concatenate all sheets along rows
        all_data = pd.concat(list(self.df.values()), axis=0)

        # Calculate the sum of all rows
        total_rows = len(all_data)
        
        print(f"Total number of rows across all sheets: {total_rows}")
    
    def train_model(self, X_train, y_train, X_test_scaled, y_test_scaled):
        # Chuyển đổi dữ liệu thành dạng phù hợp cho LSTM
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        y_train_scaled = scaler.fit_transform(np.array(y_train).reshape(-1, 1))

        # Sử dụng TimeseriesGenerator để tạo dữ liệu huấn luyện
        n_input = 10  # Định kỳ dữ liệu đầu vào
        generator = TimeseriesGenerator(X_train_scaled, y_train_scaled, length=n_input, batch_size=1)

        # Xây dựng mô hình LSTM
        model = Sequential()
        model.add(LSTM(100, activation='relu', input_shape=(n_input, X_train_scaled.shape[1])))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        # Huấn luyện mô hình
        early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        history = model.fit(generator, epochs=self.n_iterations, validation_data=(X_test_scaled, y_test_scaled), callbacks=[early_stop], verbose=1)
        
        self.model = model

    def predict(self, X_test):
        # Chuyển đổi dữ liệu kiểm tra thành dạng phù hợp cho LSTM
        X_test_scaled = self.scaler.transform(X_test)

        # Sử dụng TimeseriesGenerator để tạo dữ liệu dự đoán
        n_input = 10  # Định kỳ dữ liệu đầu vào
        generator = TimeseriesGenerator(X_test_scaled, np.zeros(len(X_test_scaled)), length=n_input, batch_size=1)

        # Dự đoán
        predictions_scaled = self.model.predict(generator)

        # Chuyển đổi dự đoán trở lại đơn vị ban đầu
        predictions = self.scaler.inverse_transform(predictions_scaled)
        return predictions

def main():
    # Set file path and other parameters
    file_path = "processed_stock_data.xlsx"
    sheet_name = None
    feature_columns = ['Open', 'High', 'Low', 'Volume', 'RSI', 'MACD_diff', 'BB_upper', 'BB_lower', 'Returns_daily', 'Volatility_daily', 'Market_Cap']

    # Initialize the LSTM object
    model = LSTM(file_path=file_path, sheet_name=sheet_name, feature_columns=feature_columns, random_state=42)
    model.analyze_data()
    model.calculate_total_rows()
    
    # Split the data
    X_train_list, X_test_list, y_train_list, y_test_list = model.split_data()

    # Combine data from lists
    X_train = pd.concat(X_train_list)
    y_train = pd.concat(y_train_list)
    X_test = pd.concat(X_test_list)
    y_test = pd.concat(y_test_list)

    # Normalization
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    y_train_scaled = scaler.fit_transform(np.array(y_train).reshape(-1, 1))
    X_test_scaled = scaler.transform(X_test)
    y_test_scaled = scaler.transform(np.array(y_test).reshape(-1, 1))

    # Train the model
    model.train_model(X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled)

    # Predict
    predictions = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R^2 Score:", r2)

if __name__ == "__main__":
    main()
