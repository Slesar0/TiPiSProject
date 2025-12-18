import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import ndcg_score
import lightgbm as lgb
from catboost import CatBoostRanker, Pool
import xgboost as xgb
import joblib

#1. Загрузка данных
df = pd.read_csv("Global_Mobile_Prices_2025_Extended.csv")
print(f"Всего объектов: {len(df)}")


#2. Предобработка и признаки
df["5g_support"] = df["5g_support"].astype(str).str.lower().map({"yes":1,"no":0}).astype(int)
month_to_num = {
    "January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
    "July":7,"August":8,"September":9,"October":10,"November":11,"December":12
}
#переводим месяца в числа
df["release_month"] = df["release_month"].astype(str).str.strip().map(month_to_num).astype(int).clip(1,12) 
#нормализуем "новизну"
df["newness"] = (df["release_month"]-1)/11.0
#делим на кварталы
df["quarter"] = ((df["release_month"]-1)//3 + 1).astype(int)

#3. Релевантность непрерывная
#нормализуем числовые признаки и добавляем в df
features_for_quality = ["price_usd","ram_gb","camera_mp","battery_mah","rating"]
scaler = MinMaxScaler()
df_norm = pd.DataFrame(scaler.fit_transform(df[features_for_quality]),
                       columns=[f"{c}_norm" for c in features_for_quality], index=df.index)
df = pd.concat([df, df_norm], axis=1)

#создаём показатель качества телефона с весами каждого признака
df["quality_score"] = 0.30*df["rating_norm"] + 0.20*df["camera_mp_norm"] + 0.15*df["battery_mah_norm"] + 0.15*df["ram_gb_norm"] + 0.20*(1-df["price_usd_norm"])

#разбиваем quality_score на 30 групп, чтобы relevance_int был целочисленным и для всех моделей были одинаковые условия
df["relevance_int"] = pd.qcut(df["quality_score"], q=30, labels=False, duplicates='drop')

# print(f"Уникальных значений quality_score: {df['quality_score'].nunique()}")
# print(f"Уникальных значений после qcut: {df['relevance_int'].nunique()}")

#4. Доп. признаки (для снижения риска переобучения и чтобы можно было более очевидно увидеть какая модель работает лучше)
#цена/качество для камеры, батареии и RAM и нормализуем
df["camera_per_price"] = df["camera_mp"]/(df["price_usd"]+1)
df["battery_per_price"] = df["battery_mah"]/(df["price_usd"]+1)
df["ram_per_price"] = df["ram_gb"]/(df["price_usd"]+1)
vfm_cols = ["camera_per_price","battery_per_price","ram_per_price"]
df[vfm_cols] = MinMaxScaler().fit_transform(df[vfm_cols])
#средний показатель технических характеристик
df["hardware_score"] = (df["ram_gb_norm"] + df["camera_mp_norm"] + df["battery_mah_norm"])/3.0

#создайм и сохраняем список признаков, используемых для ранжирования
feature_columns = ["newness","5g_support","camera_per_price","battery_per_price","ram_per_price","hardware_score"]
joblib.dump(feature_columns, "model_features.pkl")

#5. Train / Valid / Test split (1-2 / 3 / 4 кварталы)
train_df = df[df["quarter"]<=2].sort_values("quarter")
valid_df = df[df["quarter"]==3].sort_values("quarter")
test_df = df[df["quarter"]==4].sort_values("quarter")

#присваиваем разделённые данные
X_train, X_valid, X_test = train_df[feature_columns], valid_df[feature_columns], test_df[feature_columns]
#целевая переменная
y_train_int = train_df["relevance_int"].values
y_valid_int = valid_df["relevance_int"].values
y_test_int  = test_df["relevance_int"].values

#находим размеры групп
group_train = train_df.groupby("quarter").size().values
group_valid = valid_df.groupby("quarter").size().values
group_test = test_df.groupby("quarter").size().values

# вывод размеров
# print("\n Train / Valid / Test split:")
# print(f"   Train quarters: {sorted(train_df['quarter'].unique())}, size={len(train_df)}")
# print(f"   Valid quarter : {sorted(valid_df['quarter'].unique())}, size={len(valid_df)}")
# print(f"   Test quarter  : {sorted(test_df['quarter'].unique())}, size={len(test_df)}")

#6. LGBMRanker
#создаём
lgb_ranker = lgb.LGBMRanker(
    objective="lambdarank", #алгоритм ранжирования
    metric="ndcg",
    learning_rate=0.05, #шаг обучения
    num_leaves=31, #максимальное число листьев в дереве
    min_data_in_leaf=30, #минимальное число объектов в листе
    n_estimators=600, #максимальное количество деревьев
    random_state=42 #для гарантии, что модель будет одинаковой
)
#обучаем
lgb_ranker.fit(
    X_train,
    y_train_int,
    group=group_train,
    eval_set=[(X_valid, y_valid_int)], #данные для контроля качества
    eval_group=[group_valid], #группы для валидации
    eval_at=[5,10], #считаем NDCG@5 и NDCG@10
    callbacks=[lgb.early_stopping(50)] #останавливаем обучение, если метрика не улучшается 50 итераций
)

#получаем предсказание на test
y_pred_lgb = lgb_ranker.predict(X_test)


#7. CatBoostRanker
cat_ranker = CatBoostRanker(
    iterations=600, #максимальное количество деревьев
    learning_rate=0.05, #шаг обучения
    depth=6, #максимальная глубина дерева
    loss_function='YetiRank', #функция потерь для ранжирования
    eval_metric='NDCG',
    random_seed=42,
    verbose=50 #каждые 50 итераций выводим прогресс
)
#создание "пула" данных для модели (обучение)
train_pool = Pool(
    X_train,
    y_train_int,
    group_id=np.repeat( #массив в котором каждому объекту присвоен номер его группы(0-29)
        np.arange(len(group_train)),
        group_train
    )
)
#создание "пула" данных для валидации
valid_pool = Pool(
    X_valid,
    y_valid_int,
    group_id=np.repeat(
        np.arange(len(group_valid)),
        group_valid
    )
)

#обучение модели
cat_ranker.fit(
    train_pool, #обучающие данные
    eval_set=valid_pool, #данные для оценки качества на каждой итерации
    use_best_model=True #после обучения выбрать модель с лучшим NDCG на валидации
)
#предсказание на test
y_pred_cat = cat_ranker.predict(X_test)

#8. XGBoost Ranker
#преобразуем обучающие данные в формат DMatrix(специальный формат XGBoost)
dtrain = xgb.DMatrix(X_train, label=y_train_int)
dvalid = xgb.DMatrix(X_valid, label=y_valid_int)
dtest = xgb.DMatrix(X_test)

#указываем размеры групп для сравнения в одной группе
dtrain.set_group(group_train)
dvalid.set_group(group_valid)

params = {
    "objective":"rank:ndcg", #задача ранжирования
    "eval_metric":"ndcg",
    "eta":0.05, #шаг обучения
    "max_depth":6,
    "seed":42
}

xgb_ranker = xgb.train(
    params,  #параметры модели
    dtrain,
    num_boost_round=600, #максимальное количество деревьев
    evals=[(dvalid,"validation")],
    early_stopping_rounds=50, #остановка обучения, если NDCG не улучшается 50 итераций
    verbose_eval=50 #вывод прогресса каждые 50 итераций
)
#предсказание на test
y_pred_xgb = xgb_ranker.predict(dtest)

#9. NDCG по кварталам
def mean_ndcg(y_true, y_pred, groups_df, k):
    scores = []
    for q in groups_df["quarter"].unique(): #уникальные кварталы
        mask = groups_df["quarter"]==q # True там где текущий массив в цикле
        if mask.sum()>1:
            scores.append(ndcg_score([y_true[mask]], [y_pred[mask]], k=k))
    return float(np.mean(scores)) if scores else 0.0 #возвращаем средлнее по всем группам

#10. Расчёт NDCG@5, NDCG@10, NDCG@15, NDCG@20
k_values = [5, 10, 15, 20]

df_ndcg = pd.DataFrame(index=['LightGBM', 'CatBoost', 'XGBoost'])
for k in k_values:
    df_ndcg[f'NDCG@{k}'] = [
        mean_ndcg(y_test_int, y_pred_lgb, test_df[['quarter']], k=k),
        mean_ndcg(y_test_int, y_pred_cat, test_df[['quarter']], k=k),
        mean_ndcg(y_test_int, y_pred_xgb, test_df[['quarter']], k=k)
    ]

print("\nNDCG на TEST:")
print(df_ndcg)
input() #смотрим на вывод и выбираем какую модель тюнинговать

#11. Тюнинг CatBoost (т.к. он показал лучшие результаты по метрике)
best_ndcg = -1 #начальный минимум NDCG
best_params_cat = None #для хранения лучших гиперпараметров
best_cat_model = None #для хранения модель с лучими гиперпараметрами

for depth in [4,6,8]:
    for lr in [0.03,0.05,0.1]: #шаг обучения
        for iters in [300,600,900]:
            model = CatBoostRanker(
                iterations=iters,
                depth=depth,
                learning_rate=lr,
                loss_function='YetiRank',
                eval_metric='NDCG:top=10',
                random_seed=42
            )
            train_pool = Pool(X_train, y_train_int, group_id=np.repeat(np.arange(len(group_train)), group_train))
            valid_pool = Pool(X_valid, y_valid_int, group_id=np.repeat(np.arange(len(group_valid)), group_valid))
            model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
            preds = model.predict(X_valid) #предсказания для валид объектов
            score = mean_ndcg(y_valid_int, preds, valid_df[['quarter']], k=10)
            if score > best_ndcg: #обновление данных
                best_ndcg = score
                best_params_cat = (depth, lr, iters)
                best_cat_model = model

print("\nЛучшие параметры CatBoost:", best_params_cat, "NDCG@10 на VALID:", best_ndcg)

#12. Предсказание на TEST с лучшей CatBoost на VALID
y_pred_cat_best = best_cat_model.predict(X_test)
df_ndcg_best = pd.DataFrame(index=['CatBoost_tuned'])
for k in k_values:
    df_ndcg_best[f'NDCG@{k}'] = [mean_ndcg(y_test_int, y_pred_cat_best, test_df[['quarter']], k=k)]
print("\nNDCG на TEST после тюнинга CatBoost:")
print(df_ndcg_best)

#13. Cохранение модели
best_cat_model.save_model("smartphone_ranker.cbm")
print("Модель сохранена в smartphone_ranker.cbm")

