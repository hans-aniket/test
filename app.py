import streamlit as st
import tensorflow as tf
import joblib

# 1. Load & cache your models once
@st.cache_resource
def load_models():
    model = tf.keras.models.load_model('models/fruit_quality_model.keras')
    kmeans = joblib.load('models/kmeans_ripeness.pkl')

    ripeness_map = {0: 'unripe', 1: 'mid-ripe', 2: 'ripe', 3: 'overripe'}
    return model, kmeans, ripeness_map

model, kmeans, ripeness_map = load_models()

# 2. App UI
st.title("Fruit shelf life assessment")
uploaded = st.file_uploader("Upload a fruit image", type=['jpg','jpeg','png'])

if uploaded:
    # 3. Preview
    st.image(uploaded, caption="Your Image", use_column_width=True)

    # 4. Inference
    img = tf.io.decode_jpeg(uploaded.read(), channels=3)
    img = tf.image.resize(img, [300, 300])
    img = tf.expand_dims(tf.image.convert_image_dtype(img, tf.float32), 0)

    # 5. Feature extraction & clustering
    feats = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output).predict(img)
    cluster = kmeans.predict(feats)[0]
    label = ripeness_map[cluster]

    # 6. Display
    st.markdown(f"## Predicted Ripeness: **{label.title()}**")
