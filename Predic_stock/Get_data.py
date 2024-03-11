from dateutil.tz import tzlocal
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from ta.trend import macd_diff
from ta.volatility import BollingerBands
from ta.momentum import rsi
from dateutil import tz
import logging
import numpy as np
import json
import os 
from pandas.tseries.offsets import BDay
import pytz

class StockDataProcessor:
    
    def __init__(self, stock_symbols, output_file='stock_data.xlsx'):
        self.stock_symbols = stock_symbols
        self.output_file = output_file
        self.columns = ['Datetime', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'RSI', 'MACD_diff', 'BB_upper', 'BB_lower',
                    'Returns_daily',
                    'Volatility_daily', 'Market_Cap']
        self.stock_data = pd.DataFrame(columns=self.columns)
        self.scheduler = BackgroundScheduler()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def update_stock_data(self):
            self.logger.info("Updating stock data...")

            for symbol in self.stock_symbols:
                stock = yf.Ticker(symbol + '.VN')

                # Get the current time in local timezone
                current_time = datetime.now() 

                # Calculate the start time based on the current date at 9 AM (offset-naive)
                start_time = datetime.combine(current_time.date(), datetime.min.time()) + timedelta(hours=9)
                start_time = start_time.replace(tzinfo=None)

                # Calculate the end time based on the current date at 3 PM
                end_time = datetime.combine(current_time.date(), datetime.min.time()) + timedelta(hours=15)
                end_time = end_time.replace(tzinfo=None)

                # If the current time is before 9 AM, adjust the end time and start time to be the previous day
                if current_time < start_time:
                    end_time -= timedelta(days=1)
                    start_time -= timedelta(days=1)
                # If the current time is after 3 PM, adjust the end time to be the current day
                elif current_time > end_time:
                    end_time = current_time

                # Convert start_time and end_time to UTC
                start_time_utc = start_time.astimezone(tz.tzutc())
                end_time_utc = end_time.astimezone(tz.tzutc())

                # Collect data for the specified historical time range
                data = stock.history(start=start_time_utc, end=end_time_utc, interval='1m')

                if not data.empty:
                    new_data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

                    # Calculate RSI, MACD, and Bollinger Bands
                    new_data['RSI'] = rsi(new_data['Close'], window=14)
                    new_data['MACD_diff'] = macd_diff(new_data['Close'], window_slow=26, window_fast=12)
                    
                    # Bollinger Bands
                    bb = BollingerBands(new_data['Close'], window=20, window_dev=2)
                    new_data['BB_upper'] = bb.bollinger_hband()
                    new_data['BB_lower'] = bb.bollinger_lband()
                    
                    # Calculate Returns and Volatility
                    new_data['Returns_daily'] = self.calculate_returns(new_data, frequency='daily')
                    # new_data['Returns_weekly'] = self.calculate_returns(new_data, frequency='weekly')
                    # new_data['Returns_monthly'] = self.calculate_returns(new_data, frequency='monthly')

                    new_data['Volatility_daily'] = self.calculate_volatility(new_data['Returns_daily'])
                    # new_data['Volatility_weekly'] = self.calculate_volatility(new_data['Returns_weekly'])
                    # new_data['Volatility_monthly'] = self.calculate_volatility(new_data['Returns_monthly'])

                    # Calculate Market Cap and P/E Ratio
                    new_data['Market_Cap'] = self.calculate_market_cap(new_data)

                    # Explicitly set the values using .loc to avoid SettingWithCopyWarning
                    new_data.loc[:, 'Datetime'] = new_data.index.tz_convert('Asia/Ho_Chi_Minh')

                    new_data['Datetime'] = new_data['Datetime'].apply(lambda x: x.strftime('%H:%M:%S'))

                    new_data['Symbol'] = symbol
                    self.stock_data = pd.concat([self.stock_data, new_data], ignore_index=True, sort=False)

            # Use the current date as the sheet name
            name = datetime.now(tz=tzlocal())
            sheet_name = name.strftime('%Y-%m-%d')

            print(f"Sheet name: {sheet_name}")
            # Add job to scheduler only if not in trading hours
            self.scheduler.add_job(self.update_stock_data, 'interval', hours=2)

            # Write to Excel file with the current date as the sheet name
            with pd.ExcelWriter(self.output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                # Check if the sheet already exists
                if sheet_name in writer.sheets:
                    # If the sheet already exists, append the data to it
                    self.stock_data.to_excel(writer, sheet_name=sheet_name, index=False, header=True)
                else:
                    # If the sheet does not exist, create a new sheet
                    self.stock_data.to_excel(writer, sheet_name=sheet_name, index=False, header=True)

            print(f"Cập nhật thành công vào sheet '{sheet_name}' của file Excel")
            
    # def update_stock_data_500_days(self, lookback_days=500):
    #     self.logger.info("Updating stock data...")

    #     # Check if the Excel file exists
        # if not os.path.exists(self.output_file):
        #     # Create an empty DataFrame with columns
        #     initial_data = pd.DataFrame(columns=self.columns)
        #     # Write the DataFrame to Excel to create the file
        #     initial_data.to_excel(self.output_file, index=False, header=True)

    #     end_time = datetime.now()

    #     for _ in range(lookback_days):
    #         # Exclude weekends and holidays
    #         current_day = end_time - BDay(_)
    #         current_day_str = current_day.strftime('%Y-%m-%d')

    #         # Tính toán ngày 500 ngày trước (trừ ngày lễ, ngày thứ 7 và chủ nhật)
    #         day_offset = 0
    #         while day_offset < 500:
    #             previous_day = end_time - BDay(day_offset)
                
    #             # Make the datetime object tz-naive
    #             previous_day = previous_day.replace(tzinfo=None)
                
    #             previous_day_str = previous_day.strftime('%Y-%m-%d')

    #             # Kiểm tra nếu là ngày thứ 7, chủ nhật hoặc ngày nghỉ lễ
    #             if previous_day.weekday() >= 5:
    #                 day_offset += 1
    #                 continue

    #             day_offset += 1

    #         # Use pytz to localize the datetime objects
    #         start_time = pytz.timezone('Asia/Ho_Chi_Minh').localize(previous_day)
    #         end_time = pytz.timezone('Asia/Ho_Chi_Minh').localize(end_time)

    #         # Convert start_time and end_time to UTC
    #         start_time_utc = start_time.astimezone(pytz.utc)
    #         end_time_utc = end_time.astimezone(pytz.utc)

    #         sheet_name = start_time.strftime('%Y-%m-%d')

    #         for symbol in self.stock_symbols:
    #             stock = yf.Ticker(symbol + '.VN')

    #             # Convert start_time and end_time to UTC
    #             start_time_utc = start_time.astimezone(tz.tzutc())
    #             end_time_utc = end_time.astimezone(tz.tzutc())

    #             # Collect data for the specified historical time range
    #             data = stock.history(start=start_time_utc, end=end_time_utc, interval='1m')

    #             if not data.empty:
    #                 new_data = data[['Open', 'High', 'Low', 'Close', 'Volume']]

    #                 # Calculate RSI, MACD, and Bollinger Bands
    #                 new_data['RSI'] = rsi(new_data['Close'], window=14)
    #                 new_data['MACD_diff'] = macd_diff(new_data['Close'], window_slow=26, window_fast=12)

    #                 # Bollinger Bands
    #                 bb = BollingerBands(new_data['Close'], window=20, window_dev=2)
    #                 new_data['BB_upper'] = bb.bollinger_hband()
    #                 new_data['BB_lower'] = bb.bollinger_lband()

    #                 # Calculate Returns and Volatility
    #                 new_data['Returns_daily'] = self.calculate_returns(new_data, frequency='daily')
    #                 # new_data['Returns_weekly'] = self.calculate_returns(new_data, frequency='weekly')
    #                 # new_data['Returns_monthly'] = self.calculate_returns(new_data, frequency='monthly')

    #                 new_data['Volatility_daily'] = self.calculate_volatility(new_data['Returns_daily'])
    #                 # new_data['Volatility_weekly'] = self.calculate_volatility(new_data['Returns_weekly'])
    #                 # new_data['Volatility_monthly'] = self.calculate_volatility(new_data['Returns_monthly'])

    #                 # Calculate Market Cap and P/E Ratio
    #                 new_data['Market_Cap'] = self.calculate_market_cap(new_data)

    #                 # Explicitly set the values using .loc to avoid SettingWithCopyWarning
    #                 new_data.loc[:, 'Datetime'] = new_data.index.tz_convert('Asia/Ho_Chi_Minh')
    #                 new_data['Datetime'] = new_data['Datetime'].apply(lambda x: x.strftime('%H:%M:%S'))

    #                 new_data['Symbol'] = symbol
    #                 self.stock_data = pd.concat([self.stock_data, new_data], ignore_index=True, sort=False)

    #         # Write to Excel file with the current date as the sheet name
    #         with pd.ExcelWriter(self.output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    #             # If the sheet already exists, append the data to it
    #             print(f"Sheet name: {sheet_name}")

    #             self.stock_data.to_excel(writer, sheet_name=sheet_name, index=False, header=True)

    #         print(f"Cập nhật thành công vào sheet '{sheet_name}' của file Excel")

    def start_scheduler(self):
        self.scheduler.add_job(self.update_stock_data, 'interval', minutes=1)
        self.scheduler.start()

    def stop_scheduler(self):
        self.scheduler.shutdown()

    def calculate_returns(self, data, frequency='daily'):
        if frequency == 'daily':
            return data['Close'].pct_change().fillna(0)
        elif frequency == 'weekly':
            return data['Close'].resample('W').ffill().pct_change().fillna(0)
        elif frequency == 'monthly':
            return data['Close'].resample('M').ffill().pct_change().fillna(0)
        else:
            raise ValueError("Invalid frequency. Choose 'daily'.")

    def calculate_volatility(self, returns, window=20):
        return returns.rolling(window=window).std().fillna(0)

    def calculate_market_cap(self, data):
        # Assuming you have a column 'Close' for stock price and 'Volume' for shares outstanding
        data['Market_Cap'] = data['Close'] * data['Volume']
        return data['Market_Cap']

def load_stock_symbols(file_path='stock_symbols.json'):
    try:
        with open(file_path, 'r') as json_file:
            stock_symbols = json.load(json_file)
        return stock_symbols
    except FileNotFoundError:
        print(f"File '{file_path}' không tồn tại.")
        return None

def main():
    stock_symbols = load_stock_symbols()
    stock_processor = StockDataProcessor(stock_symbols)
    stock_processor.start_scheduler()

    stock_processor.update_stock_data()

if __name__ == "__main__":
    main()