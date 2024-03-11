import os
from datetime import datetime, timedelta
import random
import re
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
from datetime import datetime, timedelta

class ADALINE:
    def __init__(self, file_path, sheet_name, target_column='Close', feature_columns=None, test_size=0.2, current_date=None):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.test_size = test_size
        self.current_date = current_date or datetime.now().strftime('%Y-%m-%d')
        self.df = {}  # Initialize as an empty dictionary
        self.next_day = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        self.model = None
        self.merged_df = None
        self.X = None
        self.y = None
        self.model = None
        self.df_spliting = {}
        self.df_training = []

    def analyze_data(self):
        normalization = False

        if not self.df:
            print("Không có dữ liệu để kiểm tra.")
            return

        for sheet_name, sheet_df in self.df.items():
            missing_values = sheet_df.isnull().sum()
            nan_values = sheet_df.isna().sum()
            errors = None

            if missing_values.any():
                print(f"\nMissing Values in sheet '{sheet_name}':")
                print(missing_values)

                # Xử lý dữ liệu thiếu nếu cần
                self.handle_missing_values(sheet_name)
                
                if normalization:
                    print(f"Data in sheet '{sheet_name}' needs normalization.")
                    self.check_normalization(sheet_name)
                    self.normalize_stock_prices(sheet_name)

                if nan_values.any():
                    print(f"\nNaN Values in sheet '{sheet_name}':")
                    print(nan_values)

                    # Xử lý NaN values nếu cần
                    self.handle_nan_values(sheet_name)

                if errors:
                    print(f"\nErrors in sheet '{sheet_name}':")
                    print(errors)

                    # Xử lý errors nếu cần
                    self.handle_errors(sheet_name)            

    def check_normalization(self):
        # Kiểm tra xem dữ liệu đã được chuẩn hóa chưa
        if not self.df:
            print("Không có dữ liệu để kiểm tra.")
            return

        # Chọn các đặc trưng cần kiểm tra
        columns_to_check = ['Open', 'High', 'Low', 'Volume', 'RSI', 'MACD_diff', 'BB_upper', 'BB_lower', 'Returns_daily', 'Volatility_daily', 'Market_Cap']

        # Thống kê trung bình và độ lệch chuẩn
        stats_before_normalization = self.df[columns_to_check].describe().loc[['mean', 'std']]
        
        print("\nStatistics Before Normalization:")
        print(stats_before_normalization)

    def normalize_stock_prices(self):
        # Chọn các đặc trưng cần chuẩn hóa
        columns_to_normalize = ['Open', 'High', 'Low', 'Volume', 'RSI', 'MACD_diff', 'BB_upper', 'BB_lower', 'Returns_daily', 'Volatility_daily', 'Market_Cap']

        # Kiểm tra xem có dữ liệu để chuẩn hóa không
        if not self.df:
            print("Không có dữ liệu để chuẩn hóa.")
            return

        # Tạo một bản sao của dữ liệu để tránh ảnh hưởng đến dữ liệu gốc
        normalized_df = self.df.copy()

        # Tiêu chuẩn hóa dữ liệu sử dụng MinMaxScaler
        scaler = MinMaxScaler()
        normalized_df[columns_to_normalize] = scaler.fit_transform(normalized_df[columns_to_normalize])

        # In ra một số dòng đầu của dữ liệu đã được chuẩn hóa
        print("\nNormalized Stock Prices:")
        print(normalized_df.head())

        # Cập nhật dữ liệu trong đối tượng ADALINE
        self.df = normalized_df
    
    def remove_empty_sheets(self):
        empty_sheets = []
        for sheet_name, sheet_df in self.df.items():
            if sheet_df.empty:
                empty_sheets.append(sheet_name)
                print(f"Sheet '{sheet_name}' has no data and will be removed.")

        # Xóa các sheet chỉ có header
        if empty_sheets:
            for empty_sheet in empty_sheets:
                del self.df[empty_sheet]
            print("Empty sheets have been removed.")
    
    def handle_missing_values(self):
        # Kiểm tra và xử lý dữ liệu thiếu
        missing_columns = ['RSI', 'MACD_diff', 'BB_upper', 'BB_lower', 'Returns_daily', 'Volatility_daily', 'Market_Cap']
        missing_data = self.df[missing_columns]

        # Hiển thị dữ liệu trước khi xử lý
        print("Missing Data Before Handling:")
        print(missing_data)

        # Thay thế các giá trị thiếu bằng giá trị trung bình của cột
        self.df[missing_columns] = self.df[missing_columns].fillna(self.df[missing_columns].mean())

        # Hiển thị dữ liệu sau khi xử lý
        print("\nMissing Data After Handling:")
        print(self.df[missing_columns])

    def handle_nan_values(self):
        # Thêm logic xử lý NaN values nếu cần
        pass

    def handle_errors(self):
        # Thêm logic xử lý errors nếu cần
        pass
    
    def export_processed_data_to_excel(self, output_file_path='processed_stock_data.xlsx'):
        with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
            for sheet_name, sheet_df in self.df.items():
                # Export each sheet's data to a separate sheet in the Excel file
                sheet_df.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"\nProcessed data for sheet '{sheet_name}' has been exported.")
        
        print(f"\nAll processed data has been exported to '{output_file_path}'.")

    def split_data(self, n_splits=5, output_file_path='processed_stock_data.xlsx'):
        print(f"Dataframe before splitting data: \n{self.df}")

        # Đọc tất cả các sheet từ file Excel vào DataFrame
        xls = pd.ExcelFile(output_file_path)
        all_sheet_names = xls.sheet_names

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
        random.shuffle(valid_sheet_names)
        training_size = int(0.8 * len(valid_sheet_names))
        training_sheet_names = valid_sheet_names[:training_size]
        print(f"training_sheet_names {training_sheet_names}")
        
        testing_sheet_names = valid_sheet_names[training_size:]
        print(f"testing_sheet_names {testing_sheet_names}")
        
        training_data = pd.concat([self.df_spliting[name] for name in training_sheet_names])        
        print(f"Training Data:\n{training_data}")
        
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
    
    def train_adaline_model(self, n_splits=5):
        X_train_list, X_test_list, y_train_list, y_test_list = self.split_data(n_splits)

        # Standardize features
        scaler = StandardScaler()

        for X_train, y_train, X_test, y_test in zip(X_train_list, y_train_list, X_test_list, y_test_list):
            # Standardize features
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train Adaline model
            adaline_model = SGDRegressor(max_iter=1000, eta0=0.1, random_state=42)
            adaline_model.fit(X_train_scaled, y_train)

            # Make predictions
            y_train_pred = adaline_model.predict(X_train_scaled)
            y_test_pred = adaline_model.predict(X_test_scaled)

            # Evaluate model performance
            mse_train = mean_squared_error(y_train, y_train_pred)
            mse_test = mean_squared_error(y_test, y_test_pred)

            print(f"Mean Squared Error (Train): {mse_train}")
            print(f"Mean Squared Error (Test): {mse_test}")
 
    def fit(self, X, y, learning_rate=0.01, epochs=1000):
        # Add a column of ones to the input matrix for the bias term
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]

        # Initialize weights with zeros
        self.weights = np.zeros(X_bias.shape[1])

        for epoch in range(epochs):
            # Calculate the predicted values
            predictions = self.predict(X_bias)

            # Calculate the error
            error = y - predictions

            # Update weights using the gradient descent formula
            self.weights += learning_rate * X_bias.T.dot(error)

        print("Training complete.")
    
    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model not trained. Please use the fit method to train the model.")

        X_selected = X[self.feature_columns]

        # Add a column of ones to the input matrix for the bias term
        X_bias = np.c_[np.ones((X_selected.shape[0], 1)), X_selected]

        # Calculate the predicted values
        predictions = np.dot(X_bias, self.weights)

        return predictions

    def predict_next_day(self):
        if not self.model:
            print("No trained model available.")
            return

        # Get the latest available sheet
        latest_sheet_name = list(self.df.keys())[-1]

        # Get the latest available data
        latest_data = self.df[latest_sheet_name].tail(1)[self.feature_columns]

        # Standardize the features using the same scaler used during training
        scaler = StandardScaler()
        latest_data_scaled = scaler.fit_transform(latest_data)

        # Predict the next day's closing price using the trained model
        next_day_prediction = self.model.predict(latest_data_scaled)

        # Inverse transform the predicted value to get the original scale
        next_day_prediction_original_scale = scaler.inverse_transform(np.array([[next_day_prediction]]))

        # Print the result
        print("\nPredicted Closing Price for the Next Day:")
        print(f"Date: {self.next_day}")
        print(f"Predicted Close Price: {next_day_prediction_original_scale[0][0]}")
def main():
    # Load data from the file
    file_path = "stock_data.xlsx"
    df = pd.read_excel(file_path, sheet_name=None)  # Read all sheets
    feature_columns = ['Open', 'High', 'Low', 'Volume', 'RSI', 'MACD_diff', 'BB_upper', 'BB_lower', 'Returns_daily', 'Volatility_daily', 'Market_Cap']

    # Initialize the ADALINE object with all sheets
    model = ADALINE(file_path=file_path, sheet_name=None, feature_columns=feature_columns)

    # Check if there is data
    if df:
        for sheet_name, sheet_df in df.items():
            print(f"Handling missing values for sheet '{sheet_name}':")

            # Examine columns to handle
            columns_to_handle = ['RSI', 'MACD_diff', 'BB_upper', 'BB_lower', 'Returns_daily', 'Volatility_daily', 'Market_Cap']

            # Fill missing values with the mean
            sheet_df[columns_to_handle] = sheet_df[columns_to_handle].fillna(sheet_df[columns_to_handle].mean())

            # Save the processed data to the ADALINE object
            model.df[sheet_name] = sheet_df

            # Print processed data
            print(sheet_df.head())

        # Train the ADALINE model
        model.train_adaline_model()

        # Check if the model is trained successfully
        if model.model:
            # Predict the next day
            model.predict_next_day()
        else:
            print("Model training failed. Unable to make predictions.")

    else:
        model.remove_empty_sheets()
        print("No data available for processing.")

if __name__ == "__main__":
    main()

