import pandas as pd
import numpy as np

#%% --- 1단계: 데이터 로딩 및 구조 확인 ---

# CSV 파일을 DataFrame으로 불러옵니다.
# 'Date' 열을 인덱스로 사용하고, 날짜 형식으로 파싱합니다.
file_path = 'processed_data_1014.csv'
try:
    log_return_df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    print("✅ 데이터 로딩 성공!")
    print(f"데이터 형태 (기간, 자산 수): {log_return_df.shape}")

except FileNotFoundError:
    print(f"❌ 파일 에러: '{file_path}' 경로에 파일이 없습니다.")
    # Spyder 환경에서는 exit() 대신 raise를 사용하여 셀 실행을 중단하는 것이 더 일반적입니다.
    raise FileNotFoundError(f"'{file_path}'를 찾을 수 없습니다.")


#%% --- 2단계: Macro Factor(y)와 기타 자산(X) 분리 ---

# Macro Factor로 사용할 자산들을 리스트로 지정합니다.
# S&P500 -> SPY, DowJones -> DIA, Nasdaq -> NASDAQ 으로 데이터 컬럼명에 맞게 지정했습니다.
macro_factor_assets = ['SPY', 'NASDAQ', 'DIA', 'MSCI_World']

# 지정된 모든 자산이 데이터에 존재하는지 확인합니다.
missing_assets = [asset for asset in macro_factor_assets if asset not in log_return_df.columns]
if missing_assets:
    print(f"❌ 에러: 지정된 Macro Factor 중 다음 자산이 데이터에 없습니다: {missing_assets}")
    raise ValueError(f"'{missing_assets}'가 데이터 컬럼에 없습니다.")

# y: 여러 Macro Factor 수익률의 일별 평균을 계산하여 하나의 합성 팩터로 생성합니다.
y = log_return_df[macro_factor_assets].mean(axis=1)
y.name = 'Composite_Macro_Factor' # Series의 이름을 지정해줍니다.
# X: Macro Factor로 사용된 자산들을 모두 제외한 나머지 자산들 (독립 변수)
X = log_return_df.drop(columns=macro_factor_assets)

print("\n\n--- 변수 분리 결과 ---")
print(f" 다음 {len(macro_factor_assets)}개 자산을 통합하여 Macro Factor (y)를 생성했습니다:")
print(macro_factor_assets)
print(f"\ny의 형태: {y.shape}")
print(f"\n y를 제외 나머지 {X.shape[1]}개 자산이 X로 설정되었습니다.")
print(f"X의 형태: {X.shape}")
#%% --- 3단계: 롤링 윈도우를 이용한 일별 LASSO 자산 선택 ---
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm # 진행 상황을 시각적으로 보여주는 라이브러리
import numpy as np

# 이 셀은 1, 2단계가 포함된 'step_1_2_data_prep.py' 파일의 셀을
# 실행한 후에 실행해야 합니다. (X, y 변수 필요)
print("--- 3단계 시작: 롤링 윈도우 LASSO 변수 선택 ---")
if 'X' not in locals() or 'y' not in locals():
    raise NameError("이전 단계의 X, y 변수를 찾을 수 없습니다. step_1_2_data_prep.py를 먼저 실행해주세요.")

# --- [수정] 결측치 처리 ---
# LassoCV는 NaN 값을 처리할 수 없으므로, 0으로 채워줍니다.
X = X.fillna(0)
y = y.fillna(0)
print("결측치를 0으로 모두 채웠습니다.")


# --- 롤링 윈도우 파라미터 설정 ---
window_size = 500  # 윈도우 크기를 500일로 수정
test_start_date = '2022-01-01'
test_end_date = '2024-12-31'


# --- 결과를 저장할 변수 초기화 ---
# Key: 날짜, Value: 해당 날짜에 선택된 자산 리스트
daily_selected_assets = {}


# --- 분석 기간 설정 ---
# X의 인덱스가 날짜 형식인지 확인하고 변환
if not isinstance(X.index, pd.DatetimeIndex):
    X.index = pd.to_datetime(X.index)

# 지정된 테스트 기간에 해당하는 인덱스를 찾습니다.
try:
    start_loc = X.index.get_loc(X.index[X.index >= test_start_date][0])
    end_loc = X.index.get_loc(X.index[X.index <= test_end_date][-1]) + 1
except IndexError:
    raise ValueError("지정된 테스트 기간에 해당하는 데이터가 없습니다. 날짜를 확인해주세요.")

# 루프의 시작점은 최소한 window_size를 확보해야 합니다.
loop_start_index = max(window_size, start_loc)


# --- 롤링 윈도우 루프 실행 ---
# tqdm을 사용하여 진행률 표시
print(f"분석 기간: {X.index[loop_start_index].date()} ~ {X.index[end_loc-1].date()}")
print(f"총 {end_loc - loop_start_index}일의 기간에 대해 롤링 윈도우 분석을 시작합니다...")

for i in tqdm(range(loop_start_index, end_loc)):
    # 1. 현재 윈도우에 해당하는 데이터 슬라이싱
    current_date = X.index[i]
    window_X = X.iloc[i-window_size:i]
    window_y = y.iloc[i-window_size:i]

    # 2. 파이프라인 생성 및 학습 (매일 새로운 데이터로 재학습)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', LassoCV(cv=5, random_state=42, n_jobs=-1, tol=1e-3, max_iter=10000)) # cv=5로 조정
    ])
    pipeline.fit(window_X, window_y)

    # 3. 0이 아닌 계수를 가진 자산 선택
    coefficients = pipeline.named_steps['lasso'].coef_
    selected_assets = window_X.columns[coefficients != 0].tolist()

    # 4. 결과 기록
    daily_selected_assets[current_date] = selected_assets

print("✅ 롤링 윈도우 LASSO 자산 선택 완료!")


# --- 결과 확인 및 저장 ---
if not daily_selected_assets:
    print("\n경고: 분석 기간 동안 선택된 자산이 없습니다. 윈도우 크기나 기간 설정을 확인해주세요.")
else:
    # 딕셔너리 결과를 Pandas Series로 변환하여 다루기 쉽게 만듭니다.
    selected_assets_series = pd.Series(daily_selected_assets)
    selected_assets_series.index.name = 'Date'
    selected_assets_series.name = 'Selected_Assets'

    print("\n\n--- 일별 선택 자산 기록 (시작 5일) ---")
    print(selected_assets_series.head())

    print("\n--- 일별 선택 자산 기록 (종료 5일) ---")
    print(selected_assets_series.tail())

    # 일별 선택된 자산의 개수에 대한 통계
    selected_assets_count = selected_assets_series.apply(len)
    print("\n--- 일별 선택된 자산 개수 통계 ---")
    print(selected_assets_count.describe())

    # 결과를 CSV 파일로 저장하여 다음 단계에서 활용할 수 있습니다.
    output_path = 'daily_lasso_selected_assets_2022_2024.csv'
    selected_assets_series.to_csv(output_path, header=True)
    print(f"\n✅ 일별 선택 자산 목록이 '{output_path}' 파일로 저장되었습니다.")

#%% --- 4단계: 선택된 자산의 일별 계층적 군집 분석 ---
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from tqdm import tqdm
import ast # 문자열로 저장된 리스트를 실제 리스트로 변환하기 위함

# --- 이전 단계 결과 및 원본 데이터 로드 ---
print("--- 4단계 시작: 자산 카테고리화 (Clustering) ---")

# 3단계에서 저장한 일별 선택 자산 목록 로드
try:
    # CSV를 읽을 때, 리스트가 문자열로 저장되었으므로 converter를 사용해 실제 리스트 객체로 변환
    selected_assets_series = pd.read_csv('daily_lasso_selected_assets_2022_2024.csv',
                                         index_col='Date',
                                         parse_dates=True,
                                         converters={'Selected_Assets': ast.literal_eval}).squeeze("columns")
except FileNotFoundError:
    raise FileNotFoundError("3단계 결과 파일('daily_lasso_selected_assets_2022_2024.csv')을 찾을 수 없습니다. 3단계를 먼저 실행해주세요.")

# 1,2 단계에서 생성된 원본 수익률 데이터 X가 필요합니다.
if 'X' not in locals():
     raise NameError("원본 수익률 데이터 X를 찾을 수 없습니다. step_1_2_data_prep.py를 먼저 실행해주세요.")

# 3단계에서와 동일하게 결측치를 0으로 채워줍니다.
X = X.fillna(0)


# --- 클러스터링 파라미터 설정 ---
# 롤링 윈도우 크기는 LASSO와 동일하게 설정
window_size = 500
# 클러스터링을 위한 최소 자산 개수 (이 개수 미만이면 클러스터링 의미 없음)
min_assets_for_clustering = 3
# linkage에서 사용할 거리 기준. 'ward'는 분산 기반으로 군집을 묶어 성능이 좋음.
linkage_method = 'ward'
# fcluster에서 군집을 나눌 거리 임계값. 값에 따라 군집 개수가 달라짐.
cluster_distance_threshold = 1.0


# --- 결과를 저장할 변수 초기화 ---
# Key: 날짜, Value: {자산: 클러스터ID} 형태의 딕셔너리
daily_asset_clusters = {}


# --- 롤링 윈도우 루프 실행 ---
print(f"총 {len(selected_assets_series)}일의 기간에 대해 클러스터링을 시작합니다...")
for date, assets in tqdm(selected_assets_series.items()):
    # 1. 클러스터링을 수행할 만큼 충분한 자산이 선택되었는지 확인
    if len(assets) < min_assets_for_clustering:
        # 자산이 너무 적으면, 모두 하나의 그룹으로 간주
        daily_asset_clusters[date] = {asset: 0 for asset in assets}
        continue

    # 2. 현재 윈도우에 해당하는 데이터 슬라이싱
    # date 이전 window_size 만큼의 데이터를 사용
    try:
        window_end_loc = X.index.get_loc(date)
        window_start_loc = window_end_loc - window_size
        if window_start_loc < 0:
            continue # 충분한 윈도우 데이터가 없으면 건너뛰기

        # 선택된 자산(assets)에 대해서만 데이터 추출
        window_X_selected = X.iloc[window_start_loc:window_end_loc][assets]
    except KeyError:
        # 데이터에 날짜가 없는 경우 건너뛰기
        continue

    # 3. 상관관계 행렬 및 거리 행렬 계산
    # pdist는 데이터의 행(row)을 기준으로 거리를 계산하므로, 전치(.T)하여 자산(column) 간의 거리를 계산
    corr_matrix = window_X_selected.corr()
    # 거리 = 1 - 상관관계. 상관관계가 높을수록 거리는 가까워짐.
    # pdist는 압축된 거리 벡터를 반환
    distance_vector = pdist(corr_matrix.values, metric='correlation')

    # 4. 계층적 군집 분석 수행
    Z = linkage(distance_vector, method=linkage_method)

    # 5. 임계값을 기준으로 클러스터 형성
    # 'distance' 기준: Z의 거리값이 threshold를 넘지 않는 선에서 클러스터링
    clusters = fcluster(Z, t=cluster_distance_threshold, criterion='distance')

    # 6. 결과 기록: {자산명: 클러스터 ID} 형태로 저장
    asset_cluster_map = dict(zip(assets, clusters))
    daily_asset_clusters[date] = asset_cluster_map

print("✅ 일별 자산 클러스터링 완료!")


# --- 결과 확인 및 저장 ---
if not daily_asset_clusters:
    print("\n경고: 클러스터링 결과가 없습니다.")
else:
    clustered_assets_series = pd.Series(daily_asset_clusters)
    clustered_assets_series.index.name = 'Date'
    clustered_assets_series.name = 'Asset_Clusters'

    print("\n\n--- 일별 자산 클러스터 기록 (시작 5일) ---")
    print(clustered_assets_series.head())

    print("\n--- 일별 자산 클러스터 기록 (종료 5일) ---")
    print(clustered_assets_series.tail())

    # 결과를 CSV 파일로 저장
    output_path = 'daily_asset_clusters_2022_2024.csv'
    clustered_assets_series.to_csv(output_path, header=True)
    print(f"\n✅ 일별 자산 클러스터 목록이 '{output_path}' 파일로 저장되었습니다.")
#%% --- 5단계: 클러스터링 기반 POET 공분산 추정 ---
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import ast
import pickle # 딕셔너리 형태의 결과를 저장하고 불러오기 위함

# --- 이전 단계 결과 및 원본 데이터 로드 ---
print("--- 5단계 시작: POET 공분산 추정 (Integrated Model) ---")

try:
    # 4단계에서 저장한 일별 자산 클러스터 목록 로드
    clustered_assets_series = pd.read_csv('daily_asset_clusters_2022_2024.csv',
                                          index_col='Date',
                                          parse_dates=True,
                                          converters={'Asset_Clusters': ast.literal_eval}).squeeze("columns")
except FileNotFoundError:
    raise FileNotFoundError("4단계 결과 파일('daily_asset_clusters_2022_2024.csv')을 찾을 수 없습니다. 4단계를 먼저 실행해주세요.")

# 1,2 단계에서 생성된 원본 수익률 데이터 X가 필요합니다.
if 'X' not in locals():
     raise NameError("원본 수익률 데이터 X를 찾을 수 없습니다. step_1_2_data_prep.py를 먼저 실행해주세요.")

# 결측치를 0으로 채워줍니다.
X = X.fillna(0)


# --- POET 파라미터 설정 ---
window_size = 500
# PCA를 통해 추출할 factor의 개수. 이는 하이퍼파라미터로 조정 가능합니다.
n_factors = 2
# POET을 적용하기 위한 최소 자산 개수
min_assets_for_poet = n_factors + 1


# --- 결과를 저장할 변수 초기화 ---
# Key: 날짜, Value: 해당 날짜의 공분산 행렬(DataFrame)
daily_poet_covariances = {}


# --- 롤링 윈도우 루프 실행 ---
print(f"총 {len(clustered_assets_series)}일의 기간에 대해 POET 공분산 추정을 시작합니다...")
for date, cluster_map in tqdm(clustered_assets_series.items()):
    # 1. 클러스터링 결과로부터 현재 날짜의 자산 리스트 추출
    assets = list(cluster_map.keys())

    # 2. POET을 수행할 만큼 충분한 자산이 있는지 확인
    if len(assets) < min_assets_for_poet:
        continue

    # 3. 현재 윈도우에 해당하는 데이터 슬라이싱
    try:
        window_end_loc = X.index.get_loc(date)
        window_start_loc = window_end_loc - window_size
        if window_start_loc < 0: continue
        window_X_selected = X.iloc[window_start_loc:window_end_loc][assets]
    except KeyError:
        continue

    # --- POET 핵심 로직 ---
    # a. Factor Part 추정 (PCA)
    pca = PCA(n_components=n_factors)
    # fit_transform은 (n_samples, n_features) 데이터를 받아 (n_samples, n_factors) 형태의 factor 수익률을 반환
    factor_returns = pca.fit_transform(window_X_selected)
    # pca.components_는 (n_factors, n_features) 형태. 전치하여 (n_features, n_factors) 형태의 factor loading 행렬 생성
    loadings = pca.components_.T

    # b. Factor Part의 공분산 계산
    # rowvar=False: 열(column)을 변수로 간주하여 공분산 계산
    factor_cov = np.cov(factor_returns, rowvar=False)
    # Factor Part 공분산 = Loading * Factor_Cov * Loading.T
    cov_factor = loadings @ factor_cov @ loadings.T

    # c. Idiosyncratic Part 계산
    # 원본 데이터에서 Factor Part의 영향을 제거하여 잔차(residual)를 구함
    reconstructed_from_factors = pca.inverse_transform(factor_returns)
    residuals = window_X_selected - reconstructed_from_factors
    cov_idiosyncratic_raw = np.cov(residuals, rowvar=False)

    # d. 클러스터링 기반 Hard-Thresholding
    n_assets = len(assets)
    # 같은 클러스터에 속하면 1, 아니면 0인 마스크 행렬 생성
    threshold_mask = np.zeros((n_assets, n_assets))
    for i in range(n_assets):
        for j in range(n_assets):
            if cluster_map[assets[i]] == cluster_map[assets[j]]:
                threshold_mask[i, j] = 1.0

    # 마스크를 곱하여 다른 클러스터 간의 공분산을 0으로 만듦
    cov_idiosyncratic_thresholded = cov_idiosyncratic_raw * threshold_mask

    # e. 최종 공분산 행렬 결합
    cov_poet = cov_factor + cov_idiosyncratic_thresholded

    # 4. 결과 기록 (DataFrame 형태로 저장하여 자산 순서 정보 유지)
    cov_df = pd.DataFrame(cov_poet, index=assets, columns=assets)
    daily_poet_covariances[date] = cov_df

print("✅ 일별 POET 공분산 행렬 추정 완료!")


# --- 결과 확인 및 저장 ---
if not daily_poet_covariances:
    print("\n경고: 추정된 공분산 행렬이 없습니다.")
else:
    # 딕셔너리는 CSV로 저장하기 부적합하므로 pickle을 사용
    output_path = 'daily_integrated_model_covariances_2022_2024.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(daily_poet_covariances, f)

    print("\n\n--- 특정 날짜의 추정된 공분산 행렬 (예시) ---")
    # 마지막 날짜의 결과 출력
    last_date = list(daily_poet_covariances.keys())[-1]
    print(f"Date: {last_date.date()}")
    print(daily_poet_covariances[last_date].head())

    print(f"\n✅ 일별 공분산 행렬 딕셔너리가 '{output_path}' 파일로 저장되었습니다.")
#%% --- 6단계: 벤치마크 모델들의 공분산 행렬 추정 ---
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
import ast
import pickle

# --- 이전 단계 결과 및 원본 데이터 로드 ---
print("--- 6단계 시작: 벤치마크 모델 공분산 추정 ---")

try:
    # 3단계에서 저장한 일별 선택 자산 목록 로드
    selected_assets_series = pd.read_csv('daily_lasso_selected_assets_2022_2024.csv',
                                         index_col='Date',
                                         parse_dates=True,
                                         converters={'Selected_Assets': ast.literal_eval}).squeeze("columns")
except FileNotFoundError:
    raise FileNotFoundError("3단계 결과 파일('daily_lasso_selected_assets_2022_2024.csv')을 찾을 수 없습니다. 3단계를 먼저 실행해주세요.")

# 1,2 단계에서 생성된 원본 수익률 데이터 X가 필요합니다.
if 'X' not in locals():
     raise NameError("원본 수익률 데이터 X를 찾을 수 없습니다. step_1_2_data_prep.py를 먼저 실행해주세요.")

# 결측치를 0으로 채워줍니다.
X = X.fillna(0)


# --- 파라미터 설정 ---
window_size = 500
n_factors = 2 # POET-only 모델용 팩터 개수


# --- 결과를 저장할 변수 초기화 ---
daily_cov_lasso_only = {}
daily_cov_poet_only = {}
daily_cov_ols = {} # 'sample'에서 'ols'로 변수명 변경


# --- 롤링 윈도우 루프 실행 ---
print(f"총 {len(selected_assets_series)}일의 기간에 대해 벤치마크 공분산 추정을 시작합니다...")
all_assets = X.columns.tolist() # 전체 자산 리스트

for date, lasso_assets in tqdm(selected_assets_series.items()):
    # 1. 현재 윈도우에 해당하는 데이터 슬라이싱
    try:
        window_end_loc = X.index.get_loc(date)
        window_start_loc = window_end_loc - window_size
        if window_start_loc < 0: continue

        # 전체 자산에 대한 윈도우 데이터
        window_X_all = X.iloc[window_start_loc:window_end_loc]
        # LASSO 선택 자산에 대한 윈도우 데이터
        window_X_lasso = window_X_all[lasso_assets]

    except KeyError:
        continue

    # --- 모델별 공분산 계산 ---

    # [모델 B: LASSO-only Model]
    if len(lasso_assets) > 1:
        cov_lasso_only = window_X_lasso.cov().values
        daily_cov_lasso_only[date] = pd.DataFrame(cov_lasso_only, index=lasso_assets, columns=lasso_assets)

    # [모델 D: OLS Model (표준 공분산)] - 모델명 변경
    cov_ols = window_X_all.cov().values
    daily_cov_ols[date] = pd.DataFrame(cov_ols, index=all_assets, columns=all_assets)

    # [모델 C: POET-only Model]
    if len(all_assets) > n_factors:
        pca = PCA(n_components=n_factors)
        factor_returns = pca.fit_transform(window_X_all)
        loadings = pca.components_.T
        factor_cov = np.cov(factor_returns, rowvar=False)
        cov_factor = loadings @ factor_cov @ loadings.T

        reconstructed = pca.inverse_transform(factor_returns)
        residuals = window_X_all - reconstructed
        cov_idiosyncratic = np.diag(np.diag(np.cov(residuals, rowvar=False))) # 잔차 공분산은 대각행렬만 사용

        cov_poet_only = cov_factor + cov_idiosyncratic
        daily_cov_poet_only[date] = pd.DataFrame(cov_poet_only, index=all_assets, columns=all_assets)


print("✅ 벤치마크 모델들의 일별 공분산 행렬 추정 완료!")


# --- 결과 저장 ---
output_paths = {
    'lasso_only': 'daily_lasso_only_covariances_2022_2024.pkl',
    'poet_only': 'daily_poet_only_covariances_2022_2024.pkl',
    'ols': 'daily_ols_covariances_2022_2024.pkl' # 'sample'에서 'ols'로 키 및 파일명 변경
}

with open(output_paths['lasso_only'], 'wb') as f:
    pickle.dump(daily_cov_lasso_only, f)
print(f"✅ LASSO-only 모델 공분산이 '{output_paths['lasso_only']}' 파일로 저장되었습니다.")

with open(output_paths['poet_only'], 'wb') as f:
    pickle.dump(daily_cov_poet_only, f)
print(f"✅ POET-only 모델 공분산이 '{output_paths['poet_only']}' 파일로 저장되었습니다.")

with open(output_paths['ols'], 'wb') as f:
    pickle.dump(daily_cov_ols, f)
print(f"✅ OLS 모델 공분산이 '{output_paths['ols']}' 파일로 저장되었습니다.")

#%% --- 7단계: 포트폴리오 최적화 및 최종 성과 분석 ---
import pandas as pd
import numpy as np
import pickle
from scipy.optimize import minimize
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- 0. 이전 단계 결과 및 원본 데이터 로드 ---
print("--- 7단계 시작: 최종 성과 분석 ---")

# 1,2 단계에서 생성된 원본 수익률 데이터 X가 필요합니다.
if 'X' not in locals():
     raise NameError("원본 수익률 데이터 X를 찾을 수 없습니다. step_1_2_data_prep.py를 먼저 실행해주세요.")
X = X.fillna(0) # 결측치 처리

# 5, 6단계에서 저장한 공분산 행렬 딕셔너리 로드
model_covariances = {}
model_names = ['Integrated', 'LASSO-only', 'POET-only', 'OLS']
file_names = {
    'Integrated': 'daily_integrated_model_covariances_2022_2024.pkl',
    'LASSO-only': 'daily_lasso_only_covariances_2022_2024.pkl',
    'POET-only': 'daily_poet_only_covariances_2022_2024.pkl',
    'OLS': 'daily_ols_covariances_2022_2024.pkl'
}

for name in model_names:
    try:
        with open(file_names[name], 'rb') as f:
             model_covariances[name] = pickle.load(f)
        print(f"✅ '{file_names[name]}' 로드 완료.")
    except FileNotFoundError:
        raise FileNotFoundError(f"결과 파일 '{file_names[name]}'을 찾을 수 없습니다. 이전 단계를 먼저 실행해주세요.")


# --- 1. 포트폴리오 최적화 함수 정의 ---

def calculate_gmv_weights(cov_matrix, asset_names, gross_exposure_limit):
    """
    주어진 공분산 행렬과 제약조건 하에서 GMV 가중치를 계산합니다.
    """
    num_assets = cov_matrix.shape[0]

    # 목적 함수: 포트폴리오 분산
    def portfolio_variance(weights):
        return weights.T @ cov_matrix @ weights

    # 제약조건
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}, # Net Exposure = 1
        {'type': 'ineq', 'fun': lambda w: gross_exposure_limit - np.sum(np.abs(w))} # Gross Exposure
    ]
    # 가중치 범위 (-limit ~ +limit)
    bounds = tuple((-gross_exposure_limit, gross_exposure_limit) for _ in range(num_assets))
    # 초기 추정 가중치 (균등 배분)
    initial_weights = np.array([1/num_assets] * num_assets)

    # 최적화 실행
    result = minimize(portfolio_variance, initial_weights, method='SLSQP',
                      bounds=bounds, constraints=constraints, tol=1e-9)

    if not result.success:
        # 최적화 실패 시, 0으로 채운 가중치 반환
        return pd.Series(np.zeros(num_assets), index=asset_names)

    return pd.Series(result.x, index=asset_names)


# --- 2. 백테스팅 및 성과 분석 루프 ---

gross_exposure_levels = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
test_years = [2022, 2023, 2024]
all_results = []

print("\n백테스팅 및 성과 분석을 시작합니다...")
for year in test_years:
    print(f"\n===== 연도: {year} 분석 시작 =====")
    for limit in gross_exposure_levels:
        print(f"--- Gross Exposure Limit: {limit:.1f} ---")
        
        performance_summary = {}
        for model_name in model_names:
            
            daily_weights = {}
            daily_covs = model_covariances[model_name]
            
            # 현재 연도에 해당하는 데이터만 필터링
            yearly_covs = {date: cov for date, cov in daily_covs.items() if date.year == year}

            # 일별 최적 가중치 계산
            for date, cov_df in tqdm(yearly_covs.items(), desc=f"Optimizing for {model_name} ({year})"):
                if cov_df.empty or cov_df.shape[0] < 2: continue
                optimal_weights = calculate_gmv_weights(cov_df.values, cov_df.index, limit)
                daily_weights[date] = optimal_weights
            
            weights_series = pd.DataFrame(daily_weights).T

            # 포트폴리오 수익률 계산
            if weights_series.empty:
                portfolio_returns = pd.Series(dtype=float)
            else:
                common_dates = weights_series.index.intersection(X.index).sort_values()
                common_dates = common_dates[:-1]
                
                aligned_weights = weights_series.loc[common_dates]
                future_returns = X.shift(-1).loc[common_dates]
                
                aligned_weights, future_returns = aligned_weights.align(future_returns, join='inner', axis=1)

                portfolio_returns = (aligned_weights * future_returns).sum(axis=1)

            # 성과 지표 계산
            if not portfolio_returns.empty:
                annualized_return = portfolio_returns.mean() * 252
                annualized_risk = portfolio_returns.std() * np.sqrt(252)
                sharpe_ratio = annualized_return / annualized_risk if annualized_risk > 0 else 0
                
                performance_summary[model_name] = {
                    'Annualized Return (%)': annualized_return * 100,
                    'Annualized Risk (%)': annualized_risk * 100,
                    'Sharpe Ratio': sharpe_ratio
                }
            else:
                 performance_summary[model_name] = {key: 0 for key in ['Annualized Return (%)', 'Annualized Risk (%)', 'Sharpe Ratio']}
        
        # 결과 저장 (수정된 부분)
        results_df = pd.DataFrame(performance_summary).T.assign(
            **{'Gross Exposure': limit, 'Year': year}
        )
        all_results.append(results_df)

print("\n✅ 모든 분석이 완료되었습니다!")

# --- 3. 최종 결과 출력 ---
# 모든 결과를 하나의 데이터프레임으로 합칩니다.
final_summary_df = pd.concat(all_results)
# 인덱스였던 모델명을 'Model' 컬럼으로 변환합니다.
final_summary_df = final_summary_df.reset_index().rename(columns={'index': 'Model'})

print("\n\n--- 최종 성과 비교 요약 ---")
# 출력을 위해 임시로 멀티 인덱스를 설정하여 보기 좋게 만듭니다.
# 이 과정은 final_summary_df 원본을 변경하지 않습니다.
print(final_summary_df.set_index(['Year', 'Gross Exposure', 'Model']).to_string(float_format="%.2f"))


# --- 4. 최종 결과 시각화 ---
print("\n\n--- 최종 성과 시각화 ---")

# 시각화를 위한 데이터프레임으로 final_summary_df를 그대로 사용합니다.
# 이 데이터프레임에는 'Year', 'Model' 등이 모두 컬럼으로 존재합니다.
plot_df = final_summary_df

# 시각화할 평가지표 리스트
metrics_to_plot = ['Annualized Risk (%)', 'Annualized Return (%)', 'Sharpe Ratio']

# 각 평가지표에 대해 연도별 서브플롯 생성
for metric in metrics_to_plot:
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # relplot으로 연도별 서브플롯(facet grid) 생성
    g = sns.relplot(
        data=plot_df,
        x='Gross Exposure',
        y=metric,
        hue='Model',
        col='Year', # 이 옵션이 연도별로 플롯을 나눔
        kind='line',
        marker='o',
        markersize=8,
        linewidth=2.5,
        height=5, # 각 서브플롯의 높이
        aspect=0.6, # 각 서브플롯의 가로세로 비율
        facet_kws={'sharey': False}
    )

    # 전체 Plot에 대한 제목 및 레이블 설정
    g.fig.suptitle(f'{metric} by Gross Exposure Constraint (Yearly)', fontsize=16, fontweight='bold', y=1.03)
    g.set_axis_labels('Gross Exposure Limit', metric)
    g.set_titles("Year: {col_name}") # 각 서브플롯의 제목
    g.tight_layout(w_pad=1)
    
    plt.show()

print("\n✅ 모든 시각화가 완료되었습니다.")

