import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, f1_score
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Заголовок додатку
st.title("Training Block for Decision Support System")

st.sidebar.title("Model Training Options")

# Сесійний стан для збереження датасету
if "data" not in st.session_state:
    st.session_state.data = None

sheet_url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSwflUwlfNBeXIkQ8cmXCAv3hGyCs1w4S-TIHY3NMPFcNh1fFnHu_Nynb9I3E7Uyo59ukCDO-1xeCR8/pub?output=csv"
sheet = "https://docs.google.com/spreadsheets/d/1t1h7riadIQbvA1kPWFVRjRsIMyiowwO9YDxheRjMI3A/edit?usp=sharing"

st.markdown(
    f"""
        <a href="{sheet}" target="_blank" class="st-emotion-cache-zaw6nw e1obcldf2" style="text-decoration: none;">
             Open Data
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )
# Завантаження датасету
if st.sidebar.button("Load Dataset"):
    try:
        st.session_state.data = pd.read_csv(sheet_url, header=0)
        st.success("Dataset loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        st.session_state.data = None

# Перевірка, чи є дані
if st.session_state.data is not None:
    data = st.session_state.data
    st.write("### Dataset Preview", data.head())

    # Перевірка на пропущені значення
    if data.isnull().values.any():
        st.warning("The dataset contains missing values!")
        st.write("### Missing Value Statistics")
        st.write(data.isnull().sum())

        # Опції для обробки пропущених значень
        missing_value_option = st.sidebar.radio(
            "How would you like to handle missing values?",
            ("Remove Rows with Missing Values", "Fill Missing Values")
        )

        if missing_value_option == "Remove Rows with Missing Values":
            data = data.dropna()
            st.success("Rows with missing values have been removed.")
        elif missing_value_option == "Fill Missing Values":
            fill_method = st.sidebar.selectbox(
                "Select a method to fill missing values:",
                ["Mean", "Median", "Mode"]
            )
            for column in data.columns:
                if data[column].isnull().any():
                    if data[column].dtype in ["float64", "int64"]:
                        if fill_method == "Mean":
                            data[column].fillna(data[column].mean(), inplace=True)
                        elif fill_method == "Median":
                            data[column].fillna(data[column].median(), inplace=True)
                    else:
                        # Для категоріальних змінних використовуємо моду
                        if fill_method == "Mode":
                            data[column].fillna(data[column].mode()[0], inplace=True)
            st.success("Missing values have been filled.")


    features = st.sidebar.multiselect(
        "Select Features (X)", [col for col in data.columns if col not in ["Response", "Dt_Customer"]]
    )
    target = st.sidebar.selectbox("Select Target (y)", [col for col in data.columns if col == "Response"])


    if features and target:
        X = data[features]
        y = data[target]
        
        # Обробка категоріальних даних
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        
        # Ініціалізуємо змінну, котра буде енкодером (у випадку one-hot) або мапою енкодерів (label encoding)
        encoder_or_map = None

        if categorical_columns:
            encoder_type = st.sidebar.selectbox(
                    "Choose Encoding for Categorical Features",
                    ["One-Hot Encoding", "Label Encoding"]
                )

            if encoder_type == "One-Hot Encoding":
                one_hot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                X_encoded = one_hot_encoder.fit_transform(X[categorical_columns])
                X_encoded_df = pd.DataFrame(
                    X_encoded, 
                    columns=one_hot_encoder.get_feature_names_out(categorical_columns),
                    index=X.index
                )
                X = pd.concat([X.drop(columns=categorical_columns), X_encoded_df], axis=1)
                encoder_or_map = one_hot_encoder
            elif encoder_type == "Label Encoding":
                encoder_or_map = {}
                for col in categorical_columns:
                    label_encoder = LabelEncoder()
                    X[col] = label_encoder.fit_transform(X[col])
                    encoder_or_map[col] = label_encoder
                    
        
        feature_types = []
        for col in features:
            if col in categorical_columns:
                try:
                    pd.to_datetime(data[col])  
                    feature_types.append({"type":"date"})  
                except (ValueError, TypeError):
                    feature_types.append({"type":"categorical", "list": data[features][col].unique().tolist()})
            else:
                feature_types.append({"type":"numerical"})
        
        # Зберігаємо типи фіч в session_state
        st.session_state.feature_types = feature_types
        
        # Додавання кнопки для перегляду матриці кореляції
        if st.sidebar.button("Show Correlation Matrix"):
            if features and target:
                st.write("### Correlation Matrix for Selected Features and Target")
                selected_columns = features + [target]
                # Обчислення кореляції лише для вибраних фіч і таргета
                correlation_matrix = data[selected_columns].corr(numeric_only=True)
                # Візуалізація за допомогою Seaborn
                plt.figure(figsize=(10, 8))
                sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", square=True, cbar=True)
                plt.title("Correlation Matrix for Selected Features and Target")
                st.pyplot(plt)
            else:
                st.warning("Please select features and target before displaying the correlation matrix.")



        # Поділ на тренувальний і тестовий набори
        test_size = st.sidebar.slider("Test Size (as %)", 10, 50, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        # Вибір моделі
        model_choice = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Random Forest", "SVM", "Gradient Boosting"])
        st.sidebar.write("### Model Parameters")

        # Параметри моделі
        if model_choice == "Logistic Regression":
            C = st.sidebar.slider("Regularization Strength (C)", 0.01, 10.0, 1.0)
            solver = st.sidebar.selectbox("Solver", ["lbfgs", "liblinear", "sag", "saga"])
            model = LogisticRegression(C=C, solver=solver, random_state=42)
        elif model_choice == "Random Forest":
            n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 10, 200, 100)
            max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        elif model_choice == "SVM":
            kernel = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
            C = st.sidebar.slider("Regularization (C)", 0.1, 10.0, 1.0)
            model = SVC(kernel=kernel, C=C, random_state=42)
        elif model_choice == "Gradient Boosting":
            learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.3, 0.1)
            n_estimators = st.sidebar.slider("Number of Estimators", 50, 200, 100)
            model = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, random_state=42)


        # Тренування моделі
        if st.sidebar.button("Train Model"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            st.session_state.model = model
            st.session_state.features = features  # Зберігаємо список фіч
            st.session_state.features_types = feature_types  # Зберігаємо список фіч
            st.session_state.target = target      # Зберігаємо цільову змінну
            st.session_state.encoder_or_map = encoder_or_map
            
            st.success("Model trained and saved in session state!")

            # Метрики моделі
            st.write("### Model Metrics")
            y_pred_class = [round(pred) for pred in y_pred]
            st.write("Accuracy:", accuracy_score(y_test, y_pred_class))
            st.write("F1 Score:", f1_score(y_test, y_pred_class, average='weighted'))

          

            # Графіки
            st.write("### Actual vs Predicted")
            plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.7, color="blue")
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
            plt.xlabel("Actual")
            plt.ylabel("Predicted")
            plt.title("Actual vs Predicted")
            st.pyplot(plt)
            
         

else:
    st.write("Click 'Load Dataset' to start.")
    
