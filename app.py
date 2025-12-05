# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# Заголовок приложения
st.title("🚗 Car Price Prediction App")
st.markdown("### Predict car prices based on features with interactive EDA")

# Создаем сайдбар для навигации
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["📊 EDA Visualizations", "📈 Model Prediction", "⚖️ Model Weights"]
)

# Загрузка и предобработка данных
@st.cache_data
def load_data():
    """Загрузка и предобработка данных"""
    # Загрузка данных (аналогично вашему коду)
    df_train = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv')
    df_test = pd.read_csv('https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv')
    
    # Удаление дубликатов из train
    df_train = df_train[~df_train.drop(columns=['selling_price']).duplicated(keep='first')]
    df_train = df_train.reset_index(drop=True)
    
    # Функции для обработки признаков (из вашего кода)
    def process_features(df):
        """Обработка числовых признаков с единицами измерения"""
        def extract_value(x):
            try:
                return float(str(x).split(' ')[0]) if str(x).split(' ')[0] != '' else np.nan
            except:
                return np.nan
        
        for col in ['mileage', 'engine', 'max_power']:
            df[col] = df[col].apply(extract_value)
        
        # Обработка torque (упрощенная версия)
        def process_torque(x):
            try:
                x = str(x).replace(',', '')
                # Простая логика извлечения числа
                import re
                numbers = re.findall(r'\d+\.?\d*', x)
                return float(numbers[0]) if numbers else np.nan
            except:
                return np.nan
        
        df['torque'] = df['torque'].apply(process_torque)
        return df
    
    df_train = process_features(df_train)
    df_test = process_features(df_test)
    
    # Заполнение пропусков медианами из train
    for col in ['mileage', 'engine', 'max_power', 'torque', 'seats']:
        if col in df_train.columns:
            med = df_train[col].median()
            df_train[col] = df_train[col].fillna(med)
            df_test[col] = df_test[col].fillna(med)
    
    # Приведение типов
    df_train['engine'] = df_train['engine'].astype(int)
    df_test['engine'] = df_test['engine'].astype(int)
    df_train['seats'] = df_train['seats'].astype(int)
    df_test['seats'] = df_test['seats'].astype(int)
    
    return df_train, df_test

# Загружаем данные
df_train, df_test = load_data()

# Загрузка модели (если сохранена)
@st.cache_resource
def load_model():
    """Загрузка обученной модели"""
    try:
        # Пытаемся загрузить сохраненную модель
        with open('best_ridge_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        # Если модель не сохранена, обучаем заново
        st.warning("Предварительно обученная модель не найдена. Обучаем модель...")
        
        # Подготовка данных для обучения
        X_train = df_train.select_dtypes(exclude=['O', 'category']).copy().drop(columns=['selling_price'])
        y_train = df_train['selling_price']
        
        # Стандартизация
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Обучение модели Ridge с лучшими параметрами из вашего анализа
        model = Ridge(alpha=1)
        model.fit(X_train_scaled, y_train)
        
        # Сохраняем модель
        with open('best_ridge_model.pkl', 'wb') as f:
            pickle.dump((model, scaler, X_train.columns), f)
        
        return model, scaler, X_train.columns

# Загружаем модель
try:
    model, scaler, feature_names = load_model()
    if isinstance(model, tuple):  # Если возвращена кортеж (новая модель)
        model, scaler, feature_names = model
except:
    model, scaler, feature_names = None, None, None

# Страница 1: EDA Visualizations
if page == "📊 EDA Visualizations":
    st.header("Exploratory Data Analysis")
    
    # Выбор типа графика
    plot_type = st.selectbox(
        "Select visualization type:",
        ["Distribution Plots", "Correlation Heatmap", "Pair Plots", "Categorical Analysis", "Box Plots"]
    )
    
    # Выбор набора данных
    dataset = st.radio("Select dataset:", ["Training Data", "Test Data"], horizontal=True)
    df = df_train if dataset == "Training Data" else df_test
    
    # Выбор числовых столбцов
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_numeric = st.multiselect(
        "Select numeric features to visualize:",
        numeric_cols,
        default=numeric_cols[:5] if len(numeric_cols) > 5 else numeric_cols
    )
    
    # Выбор категориальных столбцов
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        selected_categorical = st.selectbox(
            "Select categorical feature:",
            categorical_cols
        )
    
    # Кнопка для генерации графиков
    if st.button("Generate Visualizations"):
        
        if plot_type == "Distribution Plots":
            st.subheader("Distribution of Numerical Features")
            
            n_cols = 3
            n_rows = (len(selected_numeric) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes]
            
            for idx, col in enumerate(selected_numeric):
                ax = axes[idx]
                if col in df.columns:
                    df[col].hist(ax=ax, bins=30, edgecolor='black')
                    ax.set_title(f'Distribution of {col}')
                    ax.set_xlabel(col)
                    ax.set_ylabel('Frequency')
            
            # Скрываем пустые subplots
            for idx in range(len(selected_numeric), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
            
        elif plot_type == "Correlation Heatmap":
            st.subheader("Correlation Matrix Heatmap")
            
            # Вычисляем корреляцию только для выбранных числовых признаков
            corr_df = df[selected_numeric].corr()
            
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', 
                       center=0, ax=ax, square=True, cbar_kws={"shrink": .8})
            ax.set_title('Correlation Heatmap')
            st.pyplot(fig)
            
            # Показываем топ корреляций
            st.subheader("Top Correlations")
            corr_matrix = corr_df.abs()
            np.fill_diagonal(corr_matrix.values, 0)  # Игнорируем диагональ
            
            # Получаем топ-10 корреляций
            top_corr = corr_matrix.unstack().sort_values(ascending=False).head(10)
            top_corr_df = pd.DataFrame(top_corr, columns=['Correlation']).reset_index()
            top_corr_df.columns = ['Feature 1', 'Feature 2', 'Correlation']
            st.dataframe(top_corr_df)
            
        elif plot_type == "Pair Plots":
            st.subheader("Pair Plots of Numerical Features")
            
            # Ограничиваем количество признаков для pairplot
            if len(selected_numeric) > 6:
                st.warning(f"Too many features ({len(selected_numeric)}) for pair plot. Showing first 6.")
                plot_cols = selected_numeric[:6]
            else:
                plot_cols = selected_numeric
            
            fig = sns.pairplot(df[plot_cols], diag_kind='kde', corner=False)
            st.pyplot(fig)
            
        elif plot_type == "Categorical Analysis":
            st.subheader("Categorical Feature Analysis")
            
            if selected_categorical and selected_categorical in df.columns:
                # Bar plot для категориального признака
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Count plot
                value_counts = df[selected_categorical].value_counts().head(15)
                ax1.bar(range(len(value_counts)), value_counts.values)
                ax1.set_xticks(range(len(value_counts)))
                ax1.set_xticklabels(value_counts.index, rotation=45, ha='right')
                ax1.set_title(f'Top 15 Categories in {selected_categorical}')
                ax1.set_ylabel('Count')
                
                # Box plot с целевой переменной
                if 'selling_price' in df.columns:
                    # Берем топ-10 категорий для box plot
                    top_categories = df[selected_categorical].value_counts().head(10).index
                    df_top = df[df[selected_categorical].isin(top_categories)]
                    
                    sns.boxplot(data=df_top, x=selected_categorical, y='selling_price', ax=ax2)
                    ax2.set_title(f'Selling Price by {selected_categorical}')
                    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
                
                plt.tight_layout()
                st.pyplot(fig)
                
        elif plot_type == "Box Plots":
            st.subheader("Box Plots for Numerical Features")
            
            n_cols = 2
            n_rows = (len(selected_numeric) + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes]
            
            for idx, col in enumerate(selected_numeric):
                ax = axes[idx]
                if col in df.columns:
                    df[[col]].boxplot(ax=ax)
                    ax.set_title(f'Box Plot of {col}')
                    ax.set_ylabel(col)
            
            # Скрываем пустые subplots
            for idx in range(len(selected_numeric), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # Показываем базовую статистику
    with st.expander("Show Basic Statistics"):
        st.subheader("Dataset Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Training Data Shape:**", df_train.shape)
            st.write("**Test Data Shape:**", df_test.shape)
        
        with col2:
            st.write("**Training Data Info:**")
            st.write(f"- Columns: {len(df_train.columns)}")
            st.write(f"- Numeric columns: {len(df_train.select_dtypes(include=[np.number]).columns)}")
            st.write(f"- Categorical columns: {len(df_train.select_dtypes(include=['object']).columns)}")
        
        # Показываем describe для выбранных признаков
        if selected_numeric:
            st.subheader("Descriptive Statistics")
            st.dataframe(df[selected_numeric].describe())

# Страница 2: Model Prediction
elif page == "📈 Model Prediction":
    st.header("Car Price Prediction")
    
    # Выбор способа ввода данных
    input_method = st.radio(
        "Choose input method:",
        ["📝 Manual Input", "📁 Upload CSV File"],
        horizontal=True
    )
    
    if input_method == "📝 Manual Input":
        st.subheader("Enter Car Features")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            year = st.slider("Year", 1980, 2024, 2015)
            km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
            mileage = st.number_input("Mileage (kmpl)", min_value=0.0, value=20.0)
        
        with col2:
            engine = st.number_input("Engine (CC)", min_value=0, value=1200)
            max_power = st.number_input("Max Power (bhp)", min_value=0.0, value=80.0)
            torque = st.number_input("Torque (Nm)", min_value=0.0, value=150.0)
        
        with col3:
            seats = st.selectbox("Seats", [2, 4, 5, 6, 7, 8, 9, 10])
            max_torque_rpm = st.number_input("Max Torque RPM", min_value=0, value=3000)
        
        # Категориальные признаки
        col4, col5, col6 = st.columns(3)
        
        with col4:
            fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
            seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"])
        
        with col5:
            transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
            owner = st.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
        
        # Кнопка для предсказания
        if st.button("Predict Price", type="primary"):
            if model and scaler is not None:
                try:
                    # Создаем DataFrame с введенными данными
                    input_data = pd.DataFrame({
                        'year': [year],
                        'km_driven': [km_driven],
                        'fuel': [fuel],
                        'seller_type': [seller_type],
                        'transmission': [transmission],
                        'owner': [owner],
                        'mileage': [mileage],
                        'engine': [engine],
                        'max_power': [max_power],
                        'torque': [torque],
                        'seats': [seats],
                        'max_torque_rpm': [max_torque_rpm]
                    })
                    
                    # Выбираем только числовые признаки для модели
                    X_input = input_data.select_dtypes(include=[np.number])
                    
                    # Убедимся, что порядок признаков совпадает
                    X_input = X_input.reindex(columns=feature_names, fill_value=0)
                    
                    # Масштабирование
                    X_scaled = scaler.transform(X_input)
                    
                    # Предсказание
                    prediction = model.predict(X_scaled)[0]
                    
                    # Отображаем результат
                    st.success(f"### Predicted Car Price: {prediction:,.2f}")
                    
                    # Добавляем дополнительную информацию
                    with st.expander("Show Feature Engineering"):
                        # Вычисляем дополнительные признаки
                        power_per_litre = max_power / engine if engine > 0 else 0
                        age = 2024 - year
                        km_per_year = km_driven / max(age, 1)
                        
                        st.write(f"**Power per litre:** {power_per_litre:.2f} bhp/cc")
                        st.write(f"**Car age:** {age} years")
                        st.write(f"**Kilometers per year:** {km_per_year:.0f} km/year")
                        
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
            else:
                st.error("Model not loaded properly. Please check if the model is trained.")
    
    else:  # Upload CSV File
        st.subheader("Upload CSV File for Batch Prediction")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                # Читаем CSV файл
                input_df = pd.read_csv(uploaded_file)
                st.write("**Uploaded Data Preview:**")
                st.dataframe(input_df.head())
                
                # Показываем информацию о загруженных данных
                st.write(f"**Shape:** {input_df.shape}")
                st.write(f"**Columns:** {list(input_df.columns)}")
                
                # Проверяем наличие необходимых столбцов
                required_cols = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 
                                'torque', 'seats', 'max_torque_rpm']
                
                missing_cols = [col for col in required_cols if col not in input_df.columns]
                
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                else:
                    if st.button("Predict Prices for All Cars", type="primary"):
                        if model and scaler is not None:
                            # Подготовка данных для предсказания
                            X_input = input_df.select_dtypes(include=[np.number])
                            
                            # Убедимся, что порядок признаков совпадает
                            X_input = X_input.reindex(columns=feature_names, fill_value=0)
                            
                            # Масштабирование
                            X_scaled = scaler.transform(X_input)
                            
                            # Предсказание
                            predictions = model.predict(X_scaled)
                            
                            # Добавляем предсказания к исходным данным
                            result_df = input_df.copy()
                            result_df['predicted_price'] = predictions
                            
                            # Показываем результаты
                            st.subheader("Prediction Results")
                            st.dataframe(result_df[['year', 'engine', 'max_power', 'predicted_price']].head(10))
                            
                            # Скачивание результатов
                            csv = result_df.to_csv(index=False)
                            st.download_button(
                                label="Download Predictions as CSV",
                                data=csv,
                                file_name="car_price_predictions.csv",
                                mime="text/csv"
                            )
                            
                            # Визуализация предсказаний
                            fig, ax = plt.subplots(figsize=(10, 6))
                            ax.scatter(range(len(predictions)), predictions, alpha=0.6)
                            ax.set_xlabel('Car Index')
                            ax.set_ylabel('Predicted Price')
                            ax.set_title('Distribution of Predicted Prices')
                            ax.grid(True, alpha=0.3)
                            st.pyplot(fig)
                            
                        else:
                            st.error("Model not loaded properly.")
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

# Страница 3: Model Weights
elif page == "⚖️ Model Weights":
    st.header("Model Feature Importance")
    
    if model is not None:
        # Получаем коэффициенты модели
        coefficients = model.coef_
        
        # Создаем DataFrame с важностью признаков
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        # Показываем таблицу
        st.subheader("Feature Coefficients (Sorted by Absolute Value)")
        st.dataframe(importance_df.style.format({'coefficient': '{:.4f}', 'abs_coefficient': '{:.4f}'}))
        
        # Визуализация
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Top 15 Most Important Features")
            
            fig, ax = plt.subplots(figsize=(10, 8))
            top_features = importance_df.head(15)
            colors = ['red' if x < 0 else 'green' for x in top_features['coefficient']]
            
            ax.barh(range(len(top_features)), top_features['abs_coefficient'], color=colors)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['feature'])
            ax.set_xlabel('Absolute Coefficient Value')
            ax.set_title('Top 15 Feature Importance')
            ax.invert_yaxis()  # Наиболее важные сверху
            
            # Добавляем значения коэффициентов
            for i, (coef, abs_coef) in enumerate(zip(top_features['coefficient'], top_features['abs_coefficient'])):
                ax.text(abs_coef * 1.01, i, f'{coef:.4f}', va='center')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Coefficient Distribution")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            
            # Гистограмма коэффициентов
            ax1.hist(coefficients, bins=30, edgecolor='black', alpha=0.7)
            ax1.axvline(x=0, color='r', linestyle='--', alpha=0.5)
            ax1.set_xlabel('Coefficient Value')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Distribution of Feature Coefficients')
            ax1.grid(True, alpha=0.3)
            
            # Scatter plot важности
            ax2.scatter(range(len(coefficients)), coefficients, alpha=0.6)
            ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Feature Index')
            ax2.set_ylabel('Coefficient Value')
            ax2.set_title('Feature Coefficients Scatter Plot')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Анализ признаков
        st.subheader("Feature Analysis")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.write("**Positive Impact Features** (increase price):")
            positive_features = importance_df[importance_df['coefficient'] > 0].head(5)
            for _, row in positive_features.iterrows():
                st.write(f"- **{row['feature']}**: +{row['coefficient']:.4f}")
        
        with col4:
            st.write("**Negative Impact Features** (decrease price):")
            negative_features = importance_df[importance_df['coefficient'] < 0].head(5)
            for _, row in negative_features.iterrows():
                st.write(f"- **{row['feature']}**: {row['coefficient']:.4f}")
        
        # Информация о модели
        with st.expander("Model Information"):
            st.write(f"**Model Type:** Ridge Regression")
            st.write(f"**Alpha (regularization):** {model.alpha}")
            st.write(f"**Number of Features:** {len(feature_names)}")
            st.write(f"**Model Intercept:** {model.intercept_:.2f}")
            
    else:
        st.warning("Model is not loaded. Please train or load the model first.")

# Футер приложения
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    """
    This application demonstrates car price prediction using machine learning.
    
    **Features:**
    - Interactive EDA visualizations
    - Price prediction for single or multiple cars
    - Model interpretation with feature importance
    
    **Model:** Ridge Regression with feature engineering
    """
)

# Инструкции по запуску
with st.expander("How to use this app"):
    st.markdown("""
    1. **EDA Visualizations**: Explore data distributions, correlations, and relationships
    2. **Model Prediction**: 
       - Manual Input: Enter car features manually using sliders and inputs
       - CSV Upload: Upload a CSV file with multiple cars for batch prediction
    3. **Model Weights**: Understand which features are most important for price prediction.
    """)