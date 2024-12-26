import streamlit as st
import pandas as pd
import joblib

print("model" in st.session_state , "features" in st.session_state , "feature_types" in st.session_state , "encoder" in st.session_state)
if "model" in st.session_state and "features" in st.session_state and "feature_types" in st.session_state and "encoder_or_map" in st.session_state:
    model = st.session_state.model
    features = st.session_state.features
    feature_types = st.session_state.feature_types
    encoder_or_map = st.session_state.encoder_or_map  # OneHotEncoder або словник LabelEncoder
    st.write("### Test Your Model")
    print(encoder_or_map)

    user_input = {}
    for feature, feature_type in zip(features, feature_types):
        feature_type_value = feature_type["type"]
        if feature_type_value == "categorical":
            # Якщо фіча категоріальна, використовуємо список доступних опцій
            options = feature_type.get("list", [])
            user_input[feature] = st.selectbox(f"Choose value for {feature}:", options=options)
        elif feature_type_value == "date":
            # Якщо фіча дата, додаємо вибір дати
            user_input[feature] = st.date_input(f"Select date for {feature}:")
        elif feature_type_value == "numerical":
            # Якщо фіча числова, дозволяємо вводити число
            user_input[feature] = st.number_input(f"Enter value for {feature}:", step=0.01)
        else:
            st.warning(f"Unknown feature type for {feature}")

    if st.button("Test Model"):
        # Підготовка даних для передбачення
        input_data = pd.DataFrame([user_input])

        # Обробка категоріальних даних
        print(type(encoder_or_map))
        if isinstance(encoder_or_map, dict): # LabelEncoder
            for feature in features:
                if feature in encoder_or_map:  # Перевіряємо, чи є фіча в словнику енкодера
                    input_data[feature] = encoder_or_map[feature].transform([input_data[feature][0]])
        elif hasattr(encoder_or_map, "transform"):  # OneHotEncoder
            categorical_features = [f for f, t in zip(features, feature_types) if t["type"] == "categorical"]
            print(input_data[categorical_features])
            encoded_data = pd.DataFrame(
                encoder_or_map.transform(input_data[categorical_features]),
                columns=encoder_or_map.get_feature_names_out(categorical_features)
            )
            input_data = pd.concat([input_data.drop(columns=categorical_features), encoded_data], axis=1)

        # Обробка дат
        for feature, feature_type in zip(features, feature_types):
            if feature_type["type"] == "date":
                input_data[feature] = pd.to_datetime(input_data[feature])
                input_data[feature] = (input_data[feature] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')

        # Передбачення
        prediction = model.predict(input_data)[0]
        print("predict",model.predict(input_data))
        if prediction < 0.25:
            st.warning(f"Probability of purchase is negative: {prediction}")
        elif 0.25 <= prediction < 0.50:
            st.info(f"Probability of purchase is low: {prediction}")
        elif 0.50 <= prediction < 0.8:
            st.success(f"Probability of purchase is moderate: {prediction}")
        else:
            st.success(f"Probability of purchase is high: {prediction}")

else:
    st.warning("No trained model or feature information found. Please train a model first.")
