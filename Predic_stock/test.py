import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class LSTMModel:
    def __init__(self, file_path, sheet_name, target_column='Close', feature_columns=None, test_size=0.2, current_date=None):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.test_size = test_size
        self.current_date = current_date or datetime.now().strftime('%Y-%m-%d')
        self.df = None
        self.next_day = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
        self.model = None

    def check_exist_data(self):
        # Check if the Excel file exists
        if os.path.exists(self.file_path):
            print(f"File '{self.file_path}' exists.")
            
            # Read information about sheets in the Excel file
            excel_sheets = pd.ExcelFile(self.file_path).sheet_names
            
            # Print the number of sheets and names of each sheet
            print(f"Number of sheets: {len(excel_sheets)}")
            print("List of sheets:")
            for sheet_name in excel_sheets:
                print(f"- {sheet_name}")
            
            # Check if the current date sheet exists
            if self.current_date in excel_sheets:
                print(f"\nSheet with the current date '{self.current_date}' exists in the file.")
                
                # Load and preprocess data from the current sheet
                if self.load_data():  # Assuming load_data returns True if successful
                    # Print some information about the loaded DataFrame
                    print("\nSample of loaded DataFrame:")
                    print(self.df.head())

                    # Check for consecutive days
                    consecutive_days = self.get_consecutive_days(excel_sheets)
                    
                    if consecutive_days >= 4:
                        print("Using long-term predictions for training.")
                        self.long_terms_prediction()
                    else:
                        print(f"Using short-term predictions for the next day: {self.next_day}")
                        self.short_terms_prediction()
                    
            else:
                print(f"\nNo sheet with the current date '{self.current_date}' in the file.")
        else:
            print(f"File '{self.file_path}' does not exist.")
    
    def load_data(self):
        # Implement your logic to load and preprocess data here
        # Ensure to set self.df with the loaded DataFrame
        # Return True if loading is successful, False otherwise
        try:
            # ... (your data loading and preprocessing logic)

            # Set self.df with the loaded DataFrame
            self.df = loaded_dataframe

            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def get_consecutive_days(self, sheet_names):
        # Check for consecutive days in sheet names
        sheet_dates = [datetime.strptime(date, '%Y-%m-%d') for date in sheet_names]
        consecutive_count = 0
        for i in range(len(sheet_dates) - 1):
            if (sheet_dates[i + 1] - sheet_dates[i]).days == 1:
                consecutive_count += 1
            else:
                consecutive_count = 0
        return consecutive_count
    
    def prepare_data(self):
        # Implement your data preparation logic here
        pass
    
    def long_terms_prediction(self):
        # Assuming you have a method to preprocess and prepare your data
        self.prepare_data()

        # Split the data into features (X) and target variable (y)
        X, y = self.df[self.feature_columns].values, self.df[self.target_column].values

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, shuffle=False)

        # Reshape input data to fit the LSTM model
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        # Build the LSTM model
        model = Sequential()
        model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)
        self.model = model

        # Make predictions
        predictions = self.model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, predictions)
        print(f'Mean Squared Error on Test Data: {mse}')

        # Analyze predictions and generate recommendations
        self.analyze_predictions(predictions, y_test)
        
    def short_terms_prediction(self):
        # Assuming you have a method to preprocess and prepare your data
        self.prepare_data()

        # Use the most recent data for short-term prediction
        X = self.df.tail(4)[self.feature_columns].values
        X = X.reshape((1, X.shape[0], X.shape[1]))

        # Build the LSTM model (same architecture as long-term)
        model = Sequential()
        model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model on the entire dataset
        model.fit(self.df[self.feature_columns].values.reshape((self.df.shape[0], 1, self.df.shape[1])),
                  self.df[self.target_column].values, epochs=50, batch_size=32, verbose=2)

        # Make short-term prediction
        short_term_prediction = model.predict(X)
        print(f'Short-term prediction for the next day: {short_term_prediction[0][0]}')

        # Analyze short-term prediction and generate recommendations
        self.analyze_predictions(short_term_prediction, [actual_price])  # Assuming actual_price is defined
    
    def analyze_predictions(self, predictions, actual_prices):
        # Modify this function based on your specific logic for generating recommendations
        for i in range(len(predictions)):
            predicted_price = predictions[i][0]
            actual_price = actual_prices[i]

            print(f"\nAnalysis for Day {i+1}:")
            print(f"Predicted Price: {predicted_price}")
            print(f"Actual Price: {actual_price}")

            # Generate recommendations for each day
            self.generate_recommendations(predicted_price, actual_price)

    def generate_recommendations(self, predicted_price, actual_price):
        # Modify this function based on your specific logic for generating recommendations
        if predicted_price > actual_price:
            recommendation = "SELL"
            reason = "Predicted price is higher than the actual price. Consider selling."
        elif predicted_price < actual_price:
            recommendation = "BUY"
            reason = "Predicted price is lower than the actual price. Consider buying."
        else:
            recommendation = "HOLD"
            reason = "Predicted price is similar to the actual price. Consider holding."

        print(f"Recommendation: {recommendation}")
        print(f"Reason: {reason}")

if __name__ == "__main__":
    current_date = datetime.now().strftime('%Y-%m-%d')
    file_path = "stock_data.xlsx"
    sheet_name = current_date
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'RSI', 'MACD_diff', 'BB_upper', 'BB_lower', 'Returns_daily', 'Volatility_daily', 'Market_Cap']

    # Create an instance of the LSTMModel
    lstm_model = LSTMModel(file_path, sheet_name, feature_columns=feature_columns)

    # Check for data existence and make predictions
    lstm_model.check_exist_data()
