import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler

# Загрузка данных
df = pd.read_csv("Global_Mobile_Prices_2025_Extended.csv")

# Применение one-hot к категориальным признакам
cat_cols = ["brand", "os", "processor"]
df = pd.get_dummies(df, columns=cat_cols, drop_first=False)

df["5g_support"] = df["5g_support"].str.lower().map({"yes": 1, "no": 0}).astype(int)

month_to_num = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
}

# Преобразование текстовых названий месяцев в числовые значения
df["release_month"] = df["release_month"].astype(str).str.strip().map(month_to_num)

# Приведение месяца к целому числу и ограничение диапазона от 1 до 12
df["release_month"] = df["release_month"].astype(int).clip(lower=1, upper=12)

df["newness"] = (df["release_month"] - 1) / 11.0

# Группировка месяцев по кварталам
df["quarter"] = ((df["release_month"] - 1) // 3 + 1).astype(int)

# Нормализация
features_for_quality = ["price_usd", "ram_gb", "camera_mp", "battery_mah", "rating"]
scaler = MinMaxScaler()
df_norm = scaler.fit_transform(df[features_for_quality])
df_norm = pd.DataFrame(df_norm, columns=[f"{c}_norm" for c in features_for_quality], index=df.index)
df = pd.concat([df, df_norm], axis=1)

# Расчёт целевого показателя качества
df["quality_score"] = (
    0.25 * df["rating_norm"] +
    0.15 * df["camera_mp_norm"] +
    0.20 * df["battery_mah_norm"] +
    0.15 * df["ram_gb_norm"] +
    0.20 * (1 - df["price_usd_norm"]) +  # дешевле — лучше
    0.05 * df["newness"]                 # новее — лучше
)

# Дискретизация качества в 10 уровней
df["relevance"] = pd.qcut(df["quality_score"], q=10, labels=False, duplicates='drop').astype(int)

# Сортировка данных по кварталам
df = df.sort_values("quarter")

# Формирование групп по кварталам (для обучения Ranker)
groups = df.groupby("quarter").size().tolist()
y = df["relevance"].values

# удаление вспомогательных и нечисловых столбцов
drop_cols = [
    "model", "quality_score", "relevance", "year", "release_month", "quarter",
    "price_usd_norm", "ram_gb_norm", "camera_mp_norm", "battery_mah_norm", "rating_norm", "newness"
]
X = df.drop(columns=drop_cols)

# Очистка имён признаков от недопустимых символов
X.columns = X.columns.str.replace(' ', '_').str.replace('[^A-Za-z0-9_]', '_', regex=True)

# Обучение модели ранжирования
ranker = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    learning_rate=0.05,
    num_leaves=40,
    n_estimators=300,
    min_data_in_leaf=30,
    random_state=42
)

ranker.fit(X, y, group=groups)

print("🎉 Модель успешно обучена!")
ranker.booster_.save_model("smartphone_ranker.txt")
print("📁 Модель сохранена как smartphone_ranker.txt")