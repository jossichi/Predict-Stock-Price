class ADALINE:
    def __init__(self, file_path, sheet_name, target_column='Close', feature_columns=None, test_size=0.2, current_date=None, learning_rate=0.01, n_iterations=1000, random_state=None):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.test_size = test_size
        self.current_date = current_date or datetime.now().strftime('%Y-%m-%d')
        self.df = {}  
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

                normalization = False
                if not sheet_df.empty:
                    print(f"\nHandling sheet '{sheet_name}':")

                    missing_values = sheet_df.isnull().sum()
                    nan_values = sheet_df.isna().sum()
                    errors = None

                    if missing_values.any():
                        print(f"\nMissing Values in sheet '{sheet_name}':")
                        print(missing_values)

                        # Xử lý dữ liệu thiếu nếu cần
                        self.handle_missing_values(sheet_df)

                        if normalization:
                            print(f"Data in sheet '{sheet_name}' needs normalization.")
                            self.check_normalization(sheet_name)
                            self.normalize_stock_prices(sheet_name)

                    if nan_values.any():
                        print(f"\nNaN Values in sheet '{sheet_name}':")
                        print(nan_values)

                        # Xử lý NaN values nếu cần
                        self.handle_nan_values(sheet_df)

                    if errors:
                        print(f"\nErrors in sheet '{sheet_name}':")
                        print(errors)

                        # Xử lý errors nếu cần
                        self.handle_errors(sheet_df)

                    # Update self.df with processed data
                    self.df[sheet_name] = sheet_df

            print("Tiền xử lý dữ liệu hoàn tất")
            self.export_processed_data_to_excel(output_file_path='processed_stock_data.xlsx')

        elif not self.df:
            print("Không có dữ liệu để kiểm tra.")
            return

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
    
    def handle_missing_values(self, sheet_df):
        # Kiểm tra và xử lý dữ liệu thiếu
        missing_columns = ['RSI', 'MACD_diff', 'BB_upper', 'BB_lower', 'Returns_daily', 'Volatility_daily', 'Market_Cap']
        missing_data = sheet_df[missing_columns]

        # Hiển thị dữ liệu trước khi xử lý
        print("Missing Data Before Handling:")
        print(missing_data)

        # Thay thế các giá trị thiếu bằng giá trị trung bình của cột
        sheet_df[missing_columns] = sheet_df[missing_columns].fillna(sheet_df[missing_columns].mean())

        # Hiển thị dữ liệu sau khi xử lý
        print("\nMissing Data After Handling:")
        print(sheet_df[missing_columns])

    def handle_nan_values(self,sheet_df):
        sheet_df.ffill(inplace=True)  # Sử dụng forward-fill để điền giá trị NaN từ giá trị trước đó
        sheet_df.bfill(inplace=True)  # Sử dụng backward-fill để điền giá trị NaN từ giá trị sau đó

        # Kiểm tra lại sau khi xử lý
        nan_values_after_handling = sheet_df.isna().sum()

        if nan_values_after_handling.any():
            print(f"\nNaN Values After Handling in sheet:")
            print(nan_values_after_handling)
        else:
            print(f"\nNaN Values have been successfully handled in sheet.")

    def handle_errors(self,sheet_df):
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
    