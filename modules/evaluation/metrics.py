import numpy as np
from modules.hoist import haversine
from config import WIND_BLOCK, HELI_WIND_LIMIT

def rmse(predicted, observed):
    return np.sqrt(np.mean((predicted - observed) ** 2))

def evaluate_weather_correction(kma_raw, kma_corrected, aws_observed):
    """Bias 보정 전후 RMSE 비교"""
    before = rmse(kma_raw, aws_observed)
    after = rmse(kma_corrected, aws_observed)
    print(f"보정 전 RMSE: {before:.4f} m/s")
    print(f"보정 후 RMSE: {after:.4f} m/s")
    print(f"개선율: {(before - after) / before * 100:.1f}%")
    return before, after

def evaluate_path(path, terrain, wind_field, dem_lats, dem_lons, size="heavy"):
    """경로 품질 평가

    size : "light"|"heavy" — 위험풍속 판정에 기종별 운용 제한 적용
           (소형 10 / 대형 20 m/s; 미지정 시 WIND_BLOCK 폴백)
    """
    total = len(path)
    if total == 0:
        print("경로 없음")
        return

    # 고위험 풍속 격자 통과 비율 — 기종별 운용 풍속 제한 기준
    wind_limit = HELI_WIND_LIMIT.get(size, WIND_BLOCK)
    danger_count = sum(
        1 for r, c in path if wind_field["ws"][r, c] > wind_limit
    )
    danger_rate = danger_count / total

    # 평균 경사도
    avg_slope = np.mean([terrain["slope"][r, c] for r, c in path])

    # 총 경로 거리
    total_dist = sum(
        haversine(dem_lats[path[i][0], path[i][1]],
                  dem_lons[path[i][0], path[i][1]],
                  dem_lats[path[i+1][0], path[i+1][1]],
                  dem_lons[path[i+1][0], path[i+1][1]])
        for i in range(total - 1)
    )

    print(f"총 경로 거리: {total_dist:.0f}m")
    print(f"평균 경사도: {avg_slope:.1f}°")
    print(f"위험풍속 구간 비율(>{wind_limit:.0f}m/s): {danger_rate*100:.1f}%")

    return {
        "total_distance_m": total_dist,
        "avg_slope_deg": avg_slope,
        "danger_wind_rate": danger_rate
    }

def evaluate_landing_point(predicted, actual_lat, actual_lon):
    """착륙지점 선정 오차"""
    error = haversine(predicted["latitude"], predicted["longitude"],
                      actual_lat, actual_lon)
    print(f"착륙지점 선정 오차: {error:.1f}m")
    return error