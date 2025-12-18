import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from catboost import CatBoostRanker
import joblib

# настройки страницы
st.set_page_config(
    page_title="📱 ML Ранжирование смартфонов (Learning-to-Rank)",
    page_icon="📱",
    layout="wide"
)

st.title("📱 ML-Ранжирование смартфонов (CatBoost Ranker)")
st.caption("Модель ранжирования CatBoost Ranker + пользовательские веса")

# загрузка модели
@st.cache_resource
def load_model():
    model = CatBoostRanker()
    model.load_model("smartphone_ranker.cbm")
    return model

ranker = load_model()

st.sidebar.header("⚙ Настройка весов ранжирования")

# боковая панель: настройка весов признаков
w_price = st.sidebar.slider("Вес цены (дешевле → лучше)", 0.0, 1.0, 0.15, 0.05)
w_rating = st.sidebar.slider("Вес рейтинга", 0.0, 1.0, 0.25, 0.05)
w_camera = st.sidebar.slider("Вес камеры", 0.0, 1.0, 0.20, 0.05)
w_battery = st.sidebar.slider("Вес батареи", 0.0, 1.0, 0.20, 0.05)
w_ram = st.sidebar.slider("Вес RAM", 0.0, 1.0, 0.10, 0.05)
w_newness = st.sidebar.slider("Вес новизны (месяц)", 0.0, 1.0, 0.10, 0.05)

# нормировка весов, чтобы их сумма равнялась 1
weights = np.array([w_price, w_rating, w_camera, w_battery, w_ram, w_newness])
weights /= weights.sum()
w_price, w_rating, w_camera, w_battery, w_ram, w_newness = weights

st.sidebar.caption(
    f"🎯 Итоговые веса: цена={w_price:.2f}, рейтинг={w_rating:.2f}, "
    f"камера={w_camera:.2f}, батарея={w_battery:.2f}, RAM={w_ram:.2f}, новизна={w_newness:.2f}"
)

# загрузка CSV пользователем
uploaded = st.file_uploader("📄 Загрузите CSV со смартфонами", type=["csv"])

if uploaded:
    df = pd.read_csv(uploaded)
    st.subheader("📘 Загруженные данные")
    st.dataframe(df, use_container_width=True)

    # месяц в число
    month_to_num = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
        "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
    }

    df["release_month"] = df["release_month"].astype(str).str.strip().map(month_to_num)
    df["release_month"] = df["release_month"].astype(int).clip(1, 12)

    # приведение к числу и заполнение пропусков медианой
    numeric_cols = [
        'price_usd', 'ram_gb', 'storage_gb', 'camera_mp',
        'battery_mah', 'display_size_inch', 'charging_watt', 'rating'
    ]

    if "5g_support" in df.columns:
        df["5g_support"] = df["5g_support"].astype(str).str.lower().map({
            "yes": 1, "1": 1, "true": 1,
            "no": 0, "0": 0, "false": 0
        }).fillna(0).astype(int)
    else:
        df["5g_support"] = 0

    # признак новизны
    df["newness"] = (df["release_month"] - 1) / 11.0

    # цена/качество признаки
    df["camera_per_price"] = df["camera_mp"] / (df["price_usd"] + 1)
    df["battery_per_price"] = df["battery_mah"] / (df["price_usd"] + 1)
    df["ram_per_price"] = df["ram_gb"] / (df["price_usd"] + 1)

    # нормализуем VFM признаки
    vfm_cols = ["camera_per_price", "battery_per_price", "ram_per_price"]
    df[vfm_cols] = MinMaxScaler().fit_transform(df[vfm_cols])

    # нормализуем технические характеристики
    tech_cols = ["ram_gb", "camera_mp", "battery_mah"]

    scaler_tech = MinMaxScaler()
    df_tech_norm = pd.DataFrame(
        scaler_tech.fit_transform(df[tech_cols]),
        columns=[f"{c}_norm" for c in tech_cols],
        index=df.index
    )
    df = pd.concat([df, df_tech_norm], axis=1)

    df["hardware_score"] = (df["ram_gb_norm"] + df["camera_mp_norm"] + df["battery_mah_norm"]) / 3.0

    # заполнение пропусков
    cat_cols = ["brand", "os", "processor"]
    for c in cat_cols:
        if c in df.columns:
            df[c].fillna("Unknown", inplace=True)

    # one-hot кодирование
    df_ml = pd.get_dummies(df, columns=cat_cols)

    # добавление недостающих столбцов, заполнение их нулями
    model_features = joblib.load("model_features.pkl")
    for f in model_features:
        if f not in df_ml.columns:
            df_ml[f] = 0
    X = df_ml[model_features]

    # отладочная информация
    with st.expander("🔍 Отладочная информация"):
        st.write(f"Количество объектов: {len(X)}")
        st.write(f"Используемые признаки ({len(model_features)}): {model_features}")
        key_features = ["camera_per_price", "battery_per_price", "ram_per_price", "hardware_score"]
        for feat in key_features:
            if feat in X.columns:
                st.write(f"{feat}: min={X[feat].min():.6f}, max={X[feat].max():.6f}, mean={X[feat].mean():.6f}")
    # проверка на нечисловые типы данных
    bad_cols = X.columns[X.dtypes == 'object']
    if len(bad_cols) > 0:
        st.error(f"⚠ Ошибочные типы в признаках: {list(bad_cols)}")
        st.stop()

    # прогноз модели
    l2r_scores_raw = ranker.predict(X)

    # нормализация ML Score
    scaler_ml = MinMaxScaler()
    l2r_scores = scaler_ml.fit_transform(l2r_scores_raw.reshape(-1, 1)).flatten()

    # отладка предсказаний
    with st.expander("📊 Информация о предсказаниях ML"):
        st.write(f"Нормализованные ML Scores:")
        st.write(f"Уникальных значений ML Score: {len(np.unique(l2r_scores))}")
        st.write(f"  Min: {l2r_scores.min():.6f}, Max: {l2r_scores.max():.6f}")
        st.write(f"  Mean: {l2r_scores.mean():.6f}")
        st.write(f"  Первые 5 значений: {l2r_scores[:5]}")

    # нормализация
    scaler = MinMaxScaler()
    norm_cols = ["price_usd", "rating", "camera_mp", "battery_mah", "ram_gb"]
    norm_cols = [c for c in norm_cols if c in df.columns]
    # Проверяем, не существуют ли уже *_norm столбцы для этих признаков
    user_norm_cols = [f"{c}_norm" for c in norm_cols]
    existing_user_norm_cols = [col for col in user_norm_cols if col in df.columns]

    # Удаляем существующие *_norm столбцы, если они есть (чтобы избежать дублирования)
    if existing_user_norm_cols:
        df = df.drop(columns=existing_user_norm_cols)

    if norm_cols:
        df_norm = scaler.fit_transform(df[norm_cols])
        df_norm = pd.DataFrame(
            df_norm,
            columns=[f"{c}_norm" for c in norm_cols],
            index=df.index
        )
        df = pd.concat([df, df_norm], axis=1)
    else:
        st.warning("Не найдены столбцы для нормализации пользовательского скора")

    # расчёт пользовательского скоринга
    user_score = (
        w_rating * df["rating_norm"] +
        w_camera * df["camera_mp_norm"] +
        w_battery * df["battery_mah_norm"] +
        w_ram * df["ram_gb_norm"] +
        w_newness * df["newness"] +
        w_price * (1 - df["price_usd_norm"])
    )

    # финальный скор: 60% пользовательский + 40% ML
    final_score = (0.6 * user_score + 0.4 * l2r_scores)

    df_result = df.copy()
    df_result["ML_Score"] = l2r_scores
    df_result["User_Score"] = user_score
    df_result["Final_Score"] = final_score
    df_result = df_result.sort_values("Final_Score", ascending=False).reset_index(drop=True)

    # боковая панель: фильтры
    st.sidebar.header("🔍 Фильтры")

    # по брендам
    all_brands = sorted(df_result["brand"].dropna().unique())
    selected_brands = st.sidebar.multiselect("Бренды",all_brands,default=all_brands[:5] if len(all_brands) > 5 else all_brands)

    # по операционной системе
    if "os" in df_result.columns:
        all_os = sorted(df_result["os"].dropna().unique())
        selected_os = st.sidebar.multiselect("Операционная система", all_os, default=[])
    else:
        selected_os = []

    # по процессору
    if "processor" in df_result.columns:
        all_processors = sorted(df_result["processor"].dropna().unique())
        selected_processors = st.sidebar.multiselect("Процессор", all_processors, default=[])
    else:
        selected_processors = []

    # по 5G
    st.sidebar.subheader("📶 Поддержка 5G")
    show_5g = st.sidebar.checkbox("С 5G", value=True)
    show_no_5g = st.sidebar.checkbox("Без 5G", value=True)
    allowed_5g = []
    if show_5g:
        allowed_5g.append(1)
    if show_no_5g:
        allowed_5g.append(0)
    if not allowed_5g:
        allowed_5g = [0, 1]  # показать все устройства

    # вспомогательная функция для безопасного создания слайдеров
    def safe_slider(label, col, format=None):
        min_val = float(df_result[col].min())
        max_val = float(df_result[col].max())
        if min_val == max_val:
            st.sidebar.write(f"{label}: **{min_val:.2f}**")
            return (min_val, max_val)
        else:
            return st.sidebar.slider(label, min_val, max_val, (min_val, max_val), format=format)

    # числовые фильтры
    price_range = safe_slider("Цена (USD)", "price_usd", "$%.0f")
    rating_range = safe_slider("Рейтинг", "rating", "%.2f")
    ram_range = safe_slider("RAM (ГБ)", "ram_gb", "%.0f")
    battery_range = safe_slider("Батарея (мА·ч)", "battery_mah", "%.0f")

    # применение всех фильтров
    mask = (
        df_result["price_usd"].between(price_range[0], price_range[1]) &
        df_result["rating"].between(rating_range[0], rating_range[1]) &
        df_result["ram_gb"].between(ram_range[0], ram_range[1]) &
        df_result["battery_mah"].between(battery_range[0], battery_range[1])
    )

    if selected_brands:
        mask &= df_result["brand"].isin(selected_brands)
    if "os" in df_result.columns and selected_os:
        mask &= df_result["os"].isin(selected_os)
    if "processor" in df_result.columns and selected_processors:
        mask &= df_result["processor"].isin(selected_processors)
    mask &= df_result["5g_support"].isin(allowed_5g)

    filtered_df = df_result[mask].copy()

    if filtered_df.empty:
        st.warning("Нет смартфонов, удовлетворяющих фильтрам.")
    else:
        display_cols = ["brand", "model", "os", "processor", "price_usd", "rating",
                        "camera_mp", "battery_mah", "ram_gb", "release_month",
                        "Final_Score","ML_Score", "User_Score"]
        display_df = filtered_df[display_cols].copy()

        # приведение типов для корректного отображения
        for col in ["brand", "model", "os", "processor"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].astype(str)
        for col in ["price_usd", "rating", "camera_mp", "battery_mah", "ram_gb", "Final_Score", "ML_Score", "User_Score"]:
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce')

        st.subheader(f"🏆 Результаты ({len(filtered_df)} устройств)")
        st.dataframe(display_df, use_container_width=True)

        #bar chart
        top10 = filtered_df.head(10).copy()
        top10["brand_model"] = top10["brand"] + " " + top10["model"]

        bar_chart = alt.Chart(top10).mark_bar().encode(
            x=alt.X("Final_Score:Q", title="Финальный скор"),
            y=alt.Y("brand_model:N", sort="-x", title="Смартфон"),
            color=alt.Color("brand:N", legend=alt.Legend(title="Бренд")),
            tooltip=[
                alt.Tooltip("brand:N", title="Бренд"),
                alt.Tooltip("model:N", title="Модель"),
                alt.Tooltip("release_month:N", title="Месяц"),
                alt.Tooltip("price_usd:Q", title="Цена, $", format="$.0f"),
                alt.Tooltip("ram_gb:Q", title="RAM, ГБ", format=".0f"),
                alt.Tooltip("camera_mp:Q", title="Камера, МП", format=".0f"),
                alt.Tooltip("battery_mah:Q", title="Батарея, мА·ч", format=".0f"),
                alt.Tooltip("Final_Score:Q", title="Итоговый скор", format=".3f")
            ]
        ).properties(
            title="ТОП-10 смартфонов по финальному скору",
            height=500
        ).interactive()

        st.altair_chart(bar_chart, use_container_width=True)

        # сравнение ML Score и User Score
        scatter = alt.Chart(filtered_df).mark_circle(size=60).encode(
            x=alt.X("User_Score:Q", title="User Score"),
            y=alt.Y("ML_Score:Q", title="ML Score"),
            color=alt.Color("brand:N", legend=None),
            tooltip=["brand", "model", "User_Score", "ML_Score", "Final_Score"]
        ).properties(
            title="Сравнение: ML Score vs User Score",
            width=600,
            height=400
        ).interactive()

        st.altair_chart(scatter, use_container_width=True)

        # распределение оценок
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Средний ML Score", f"{filtered_df['ML_Score'].mean():.3f}")
        with col2:
            st.metric("Средний User Score", f"{filtered_df['User_Score'].mean():.3f}")
        with col3:
            st.metric("Средний Final Score", f"{filtered_df['Final_Score'].mean():.3f}")
    # кнопка для скачивания результатов
    csv = df_result.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Скачать полный результат (CSV)",
        data=csv,
        file_name="ranked_smartphones.csv",
        mime="text/csv",
        use_container_width=True
    )

else:
    st.info("⬆ Загрузите CSV файл, чтобы продолжить")
