"""
data/loader.py
统一数据集加载器 —— 所有数据集通过同一接口返回 (X_iter, y_iter) 的生成器
当前支持：US_Accidents / Electricity / KDDCup99 / CoverType / Airlines / INSECTS
          / Phishing / Bananas / ImageSegments
"""

import pandas as pd
import numpy as np
import random
from pathlib import Path
from sklearn.datasets import fetch_kddcup99, fetch_covtype
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_DIR = Path(__file__).parent  # data/ 目录
US_ACCIDENTS_RAW_PATH = DATA_DIR / 'US_Accidents_March23.csv'
US_ACCIDENTS_READY_PATH = DATA_DIR / 'accidents_model_ready.csv'
ELECTRICITY_PATH = DATA_DIR / 'electricity.csv'
KDDCUP99_PATH = DATA_DIR / 'kddcup99_10_data.gz'
COVTYPE_PATH = DATA_DIR / 'covtype.data.gz'
INSECTS_PATH = DATA_DIR / 'INSECTS-abrupt_balanced_norm.csv'

KDDCUP99_COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins',
    'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
    'num_file_creations', 'num_shells', 'num_access_files',
    'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
    'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'labels',
]

COVTYPE_COLUMNS = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',
    'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
    'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
    'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10',
    'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14',
    'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18',
    'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22',
    'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',
    'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
    'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
    'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38',
    'Soil_Type39', 'Soil_Type40',
]


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def _encode_dataframe(df: pd.DataFrame):
    """将 DataFrame 中所有 object 列 LabelEncode，数值列不动"""
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


def _to_stream(X: pd.DataFrame, y: pd.Series):
    """把 DataFrame/Series 变成 River 风格的 (dict, label) 迭代器"""
    for i in range(len(X)):
        yield X.iloc[i].to_dict(), y.iloc[i]


def _minority_classes(y: pd.Series):
    counts = y.value_counts()
    avg = counts.mean()
    minority = counts[counts < avg].index.tolist()
    if not minority:
        minority = [counts.idxmin()]
    return [int(c) for c in minority]


def _load_river_dataset(
    dataset_cls,
    name: str,
    seed: int = 42,
    shuffle: bool = False,
    minority_classes: list = None,
    class_names: list = None,
):
    dataset = dataset_cls()
    data = list(dataset)
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(data)

    X = pd.DataFrame([x for x, _ in data])
    X = _encode_dataframe(X)

    le = LabelEncoder()
    y = pd.Series(le.fit_transform([label for _, label in data]))

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    info = {
        'name': name,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_classes': int(y.nunique()),
        'minority_classes': minority_classes or _minority_classes(y),
        'class_names': class_names or [str(c) for c in le.classes_],
    }
    return _to_stream(X_train, y_train), _to_stream(X_test, y_test), info


def _prepare_us_accidents(
    raw_path: Path = US_ACCIDENTS_RAW_PATH,
    output_path: Path = US_ACCIDENTS_READY_PATH,
    sample_rate: float = 0.05,
    seed: int = 42,
) -> Path:
    rng = random.Random(seed)
    df = pd.read_csv(
        raw_path,
        skiprows=lambda row: row > 0 and rng.random() > sample_rate,
    )

    cols_to_drop = [
        'End_Lat', 'End_Lng', 'Wind_Chill(F)', 'Precipitation(in)',
        'Description', 'Street', 'Zipcode', 'Airport_Code',
        'ID', 'Country', 'Source',
    ]
    df = df.drop(columns=cols_to_drop, errors='ignore')

    numeric_cols = [
        'Temperature(F)', 'Humidity(%)', 'Visibility(mi)',
        'Pressure(in)', 'Wind_Speed(mph)',
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    text_cols = [
        'Weather_Condition', 'Wind_Direction', 'Weather_Timestamp',
        'City', 'Timezone', 'Sunrise_Sunset', 'Civil_Twilight',
        'Nautical_Twilight', 'Astronomical_Twilight',
    ]
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed')
    df = df.sort_values('Start_Time').reset_index(drop=True)
    df['Hour'] = df['Start_Time'].dt.hour
    df['Weekday'] = df['Start_Time'].dt.dayofweek
    df['Month'] = df['Start_Time'].dt.month

    df = df.drop(columns=['Start_Time', 'End_Time', 'Weather_Timestamp'], errors='ignore')
    df = df.drop(columns=['County', 'City', 'Weather_Condition'], errors='ignore')

    day_night_cols = [
        'Sunrise_Sunset', 'Civil_Twilight',
        'Nautical_Twilight', 'Astronomical_Twilight',
    ]
    for col in day_night_cols:
        if col in df.columns:
            df[col] = df[col].map({'Day': 1, 'Night': 0})

    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)

    onehot_cols = [c for c in ['Wind_Direction', 'State', 'Timezone'] if c in df.columns]
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


# ─────────────────────────────────────────────
# 各数据集加载函数
# ─────────────────────────────────────────────

def load_us_accidents(path: str = None, sample_frac: float = 1.0, seed: int = 42):
    """
    加载 US Accidents 数据集（你已有的 accidents_model_ready.csv）
    返回：(stream_generator, minority_classes, n_classes, dataset_info)
    """
    if path is None:
        path = US_ACCIDENTS_READY_PATH
        if not path.exists() and US_ACCIDENTS_RAW_PATH.exists():
            path = _prepare_us_accidents(seed=seed)

    if not Path(path).exists():
        raise FileNotFoundError(
            f"US Accidents data not found: {path}\n"
            "Place accidents_model_ready.csv in data/, or place "
            "US_Accidents_March23.csv in data/ so it can be prepared automatically."
        )

    df = pd.read_csv(path)
    df = _encode_dataframe(df)

    X = df.drop(columns=['Severity'])
    y = df['Severity'] - 1  # 转为 0,1,2,3

    if sample_frac < 1.0:
        df_sample = pd.concat([X, y], axis=1).sample(frac=sample_frac, random_state=seed)
        X = df_sample.drop(columns=['Severity'])
        y = df_sample['Severity']

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    info = {
        'name': 'US_Accidents',
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_classes': 4,
        'minority_classes': [0, 2, 3],
        'class_names': ['Sev1', 'Sev2', 'Sev3', 'Sev4'],
    }
    return _to_stream(X_train, y_train), _to_stream(X_test, y_test), info


def load_electricity(seed: int = 42):
    """
    Electricity Market 数据集（River 内置，二分类，含概念漂移）
    标签：0 = UP, 1 = DOWN
    """
    if ELECTRICITY_PATH.exists():
        df = pd.read_csv(ELECTRICITY_PATH)
        X = df.drop(columns=['class'])
        y = df['class'].map({'UP': 0, 'DOWN': 1})

        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        info = {
            'name': 'Electricity',
            'n_samples': len(X),
            'n_features': X.shape[1],
            'n_classes': 2,
            'minority_classes': [1],
            'class_names': ['UP', 'DOWN'],
        }
        return _to_stream(X_train, y_train), _to_stream(X_test, y_test), info

    try:
        from river.datasets import Elec2
        dataset = Elec2()
        data = list(dataset)
        random_state = np.random.RandomState(seed)
        random_state.shuffle(data)

        split = int(len(data) * 0.8)
        train_data = data[:split]
        test_data = data[split:]

        def make_stream(d):
            for x, y in d:
                yield x, int(y)

        info = {
            'name': 'Electricity',
            'n_samples': len(data),
            'n_features': 8,
            'n_classes': 2,
            'minority_classes': [1],
            'class_names': ['UP', 'DOWN'],
        }
        return make_stream(train_data), make_stream(test_data), info

    except ImportError:
        raise ImportError("请先安装: pip install river")


def load_kddcup99(seed: int = 42):
    """
    KDD Cup 99（10% 子集），极不平衡多分类
    只保留数量最多的5个攻击类型，简化为5类问题
    """
    if KDDCUP99_PATH.exists():
        df = pd.read_csv(KDDCUP99_PATH, compression='gzip', header=None, names=KDDCUP99_COLUMNS)
        X = df.drop(columns=['labels'])
        y_raw = df['labels']
    else:
        data = fetch_kddcup99(percent10=True, random_state=seed, as_frame=True)
        X, y_raw = data.data, data.target

    # 只保留出现频率最高的5个类别
    top5 = pd.Series(y_raw).value_counts().head(5).index.tolist()
    mask = y_raw.isin(top5)
    X, y_raw = X[mask].reset_index(drop=True), y_raw[mask].reset_index(drop=True)

    # 编码标签为整数
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_raw))
    X = _encode_dataframe(X)

    # 标准化数值特征
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # 找少数类（出现频率 < 平均）
    counts = y.value_counts()
    avg = counts.mean()
    minority = counts[counts < avg].index.tolist()

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    info = {
        'name': 'KDDCup99',
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_classes': 5,
        'minority_classes': minority,
        'class_names': [str(c) for c in le.classes_[:5]],
    }
    return _to_stream(X_train, y_train), _to_stream(X_test, y_test), info


def load_covertype(seed: int = 42):
    """
    Forest CoverType，581,012 条，7 类，典型多类不平衡
    """
    if COVTYPE_PATH.exists():
        df = pd.read_csv(COVTYPE_PATH, compression='gzip', header=None, names=COVTYPE_COLUMNS + ['Cover_Type'])
        X = df.drop(columns=['Cover_Type'])
        y_raw = df['Cover_Type'] - 1  # 转为 0-6
    else:
        data = fetch_covtype(as_frame=True)
        X, y_raw = data.data, data.target - 1  # 转为 0-6

    # 打乱（无时序约束，作为静态不平衡基准测试）
    idx = np.random.RandomState(seed).permutation(len(X))
    X = X.iloc[idx].reset_index(drop=True)
    y = pd.Series(y_raw.values[idx])

    counts = y.value_counts()
    avg = counts.mean()
    minority = counts[counts < avg].index.tolist()

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    info = {
        'name': 'CoverType',
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_classes': 7,
        'minority_classes': minority,
        'class_names': [f'Type{i+1}' for i in range(7)],
    }
    return _to_stream(X_train, y_train), _to_stream(X_test, y_test), info


def load_airlines(path: str = None, seed: int = 42):
    """
    Airlines 数据集。
    项目中已提供 data/airlines.csv。
    若需要重新生成，可从 OpenML 的 airlines ARFF 源文件下载后转为 CSV。
    """
    if path is None:
        path = DATA_DIR / 'airlines.csv'

    if not Path(path).exists():
        raise FileNotFoundError(
            f"Airlines 数据集未找到：{path}\n"
            "请先准备 data/airlines.csv，或从 OpenML 官方 ARFF 源文件转换生成。"
        )

    df = pd.read_csv(path)
    df = _encode_dataframe(df)

    # 最后一列通常是标签
    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y_raw = df[target_col]
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y_raw))

    # 时序打乱（Airlines 本身无严格时序）
    idx = np.random.RandomState(seed).permutation(len(X))
    X = X.iloc[idx].reset_index(drop=True)
    y = pd.Series(y.values[idx])

    counts = y.value_counts()
    avg = counts.mean()
    minority = counts[counts < avg].index.tolist()

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    info = {
        'name': 'Airlines',
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_classes': int(y.nunique()),
        'minority_classes': minority,
        'class_names': [str(c) for c in le.classes_],
    }
    return _to_stream(X_train, y_train), _to_stream(X_test, y_test), info


def load_insects(path: str = None):
    """
    INSECTS abrupt balanced 数据集。
    当前项目落地为 CSV：data/INSECTS-abrupt_balanced_norm.csv
    """
    if path is None:
        path = INSECTS_PATH

    if not Path(path).exists():
        raise FileNotFoundError(
            f"INSECTS 数据集未找到：{path}\n"
            "请先准备 data/INSECTS-abrupt_balanced_norm.csv。"
        )

    df = pd.read_csv(path)
    X = df.drop(columns=['Class'])
    y = df['Class'].astype(int)

    counts = y.value_counts()
    avg = counts.mean()
    minority = counts[counts < avg].index.tolist()

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    info = {
        'name': 'INSECTS',
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_classes': int(y.nunique()),
        'minority_classes': minority,
        'class_names': [str(c) for c in sorted(y.unique())],
    }
    return _to_stream(X_train, y_train), _to_stream(X_test, y_test), info


def load_phishing(seed: int = 42):
    from river.datasets import Phishing
    return _load_river_dataset(
        Phishing,
        'Phishing',
        seed=seed,
        minority_classes=[1],
        class_names=['Legitimate', 'Phishing'],
    )


def load_bananas(seed: int = 42):
    from river.datasets import Bananas
    return _load_river_dataset(
        Bananas,
        'Bananas',
        seed=seed,
        minority_classes=[1],
        class_names=['False', 'True'],
    )


def load_image_segments(seed: int = 42):
    from river.datasets import ImageSegments
    return _load_river_dataset(
        ImageSegments,
        'ImageSegments',
        seed=seed,
        shuffle=True,
        minority_classes=[1, 2, 3],
        class_names=[str(i) for i in range(7)],
    )


# ─────────────────────────────────────────────
# 统一注册表 —— 在 run_all.py 中用名字调用
# ─────────────────────────────────────────────
DATASET_REGISTRY = {
    'US_Accidents': load_us_accidents,
    'Electricity':  load_electricity,
    'KDDCup99':     load_kddcup99,
    'CoverType':    load_covertype,
    'Airlines':     load_airlines,
    'INSECTS':      load_insects,
    'Phishing':      load_phishing,
    'Bananas':       load_bananas,
    'ImageSegments': load_image_segments,
}
