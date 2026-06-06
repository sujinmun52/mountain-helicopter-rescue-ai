"""
실시간 동적 경로 최적화 시스템
- 기상 임계치 밴드: 풍속 임계치 이하 시 경로 유지
- 진행 방향 가중치: 구조대원 진행 방향 -15% 할인
- 경로 변경 판단: 위험도 증가율 25% 초과 시에만 재계산
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional
from datetime import datetime


@dataclass
class PathUpdateConfig:
    """경로 업데이트 설정"""
    # 기상 임계치 밴드
    wind_threshold_m_s: float = 12.0  # m/s - 이 이상이면 경로 변경 고려
    wind_threshold_band: float = 1.0  # m/s - 임계치 ±범위 (이력현상)
    
    # 위험도 임계치
    risk_increase_threshold: float = 0.25  # 25% 이상 증가 시 경로 재계산
    
    # 진행 방향 가중치
    direction_weight_discount: float = 0.15  # 15% 할인
    direction_lookahead: int = 3  # 향후 3개 웨이포인트까지만 할인
    
    # 화면 업데이트
    min_update_interval_minutes: int = 10  # 최소 10분 간격
    soft_update_enabled: bool = True  # 소프트 업데이트 활성화
    
    # 관제 승인
    require_approval: bool = True  # 경로 변경 시 승인 필요


class RealTimePathOptimizer:
    """실시간 경로 최적화 엔진"""
    
    def __init__(self, config: PathUpdateConfig = None):
        self.config = config or PathUpdateConfig()
        self.current_path: Optional[List[Tuple[int, int]]] = None
        self.current_risk_score: float = 0.0
        self.last_update_time: Optional[datetime] = None
        self.current_waypoint_idx: int = 0  # 구조대원 현재 위치 (웨이포인트 인덱스)
        self.approval_pending: bool = False
        self.pending_new_path: Optional[List[Tuple[int, int]]] = None
        self.wind_locked: bool = False  # 풍속 로킹 상태
        
    def evaluate_path_risk(self, 
                          path: List[Tuple[int, int]], 
                          terrain: Dict,
                          wind_field: np.ndarray,
                          dem_lats: np.ndarray,
                          dem_lons: np.ndarray) -> float:
        """
        경로의 위험도 점수 계산
        
        Args:
            path: 웨이포인트 좌표 리스트 [(row, col), ...]
            terrain: 지형 데이터 딕셔너리
            wind_field: 풍속 필드
            dem_lats, dem_lons: 위경도 배열
            
        Returns:
            위험도 점수 (0~1)
        """
        if not path or len(path) < 2:
            return 0.0
        
        total_risk = 0.0
        segment_count = len(path) - 1
        
        for i in range(segment_count):
            row, col = path[i]
            
            # 경계 체크
            if not (0 <= row < wind_field.shape[0] and 0 <= col < wind_field.shape[1]):
                continue
            
            # 풍속 위험도
            wind_speed = wind_field[int(row), int(col)]
            wind_risk = min(wind_speed / 20.0, 1.0)  # 20m/s를 기준으로 정규화
            
            # 경사도 위험도
            slope = terrain.get('slope', np.zeros_like(wind_field))[int(row), int(col)]
            slope_risk = min(slope / 90.0, 1.0)  # 90도를 기준으로 정규화
            
            # 숲 위험도
            is_open = terrain.get('is_open', np.ones_like(wind_field))[int(row), int(col)]
            forest_risk = 0.0 if is_open else 0.3
            
            # 능선 위험도
            is_ridge = terrain.get('is_ridge', np.zeros_like(wind_field))[int(row), int(col)]
            ridge_risk = 0.2 if is_ridge else 0.0
            
            # 가중 평균 (slope 우선: 44° 절벽 회피, wind 완화)
            segment_risk = (wind_risk * 0.2 + slope_risk * 0.5 +
                           forest_risk * 0.2 + ridge_risk * 0.1)
            total_risk += segment_risk
        
        return total_risk / segment_count if segment_count > 0 else 0.0
    
    def should_update_path(self, 
                          current_path: List[Tuple[int, int]],
                          new_path: List[Tuple[int, int]],
                          wind_field: np.ndarray,
                          terrain: Dict,
                          dem_lats: np.ndarray,
                          dem_lons: np.ndarray) -> Tuple[bool, str]:
        """
        경로 변경 필요 여부 판단
        
        Returns:
            (should_update: bool, reason: str)
        """
        # 1. 풍속 임계치 밴드 체크
        current_max_wind = self._get_max_wind_on_path(current_path, wind_field)
        
        # 기본 임계치 (처음 락 될 때)
        threshold = self.config.wind_threshold_m_s
        
        # 이력현상: 이미 락된 상태면 임계치 + 밴드
        if self.wind_locked:
            threshold += self.config.wind_threshold_band
        
        if current_max_wind <= threshold:
            self.wind_locked = True
            return False, f"풍속 {current_max_wind:.1f}m/s ≤ {threshold:.1f}m/s (경로 잠금)"
        
        self.wind_locked = False
        
        # 2. 위험도 증가율 체크
        current_risk = self.evaluate_path_risk(current_path, terrain, wind_field, 
                                               dem_lats, dem_lons)
        new_risk = self.evaluate_path_risk(new_path, terrain, wind_field, 
                                          dem_lats, dem_lons)
        
        risk_increase_rate = (current_risk - new_risk) / current_risk if current_risk > 0 else 0
        
        # 새 경로가 25% 이상 안전하면 변경
        if risk_increase_rate >= self.config.risk_increase_threshold:
            return True, f"위험도 {risk_increase_rate*100:.1f}% 감소 (임계치: 25%)"
        
        return False, f"위험도 감소 {risk_increase_rate*100:.1f}% < 25% (변경 안함)"
    
    def apply_direction_weight(self, 
                              path: List[Tuple[int, int]],
                              current_waypoint_idx: int,
                              penalty_field: np.ndarray) -> np.ndarray:
        """
        구조대원 진행 방향에 가중치 할인 적용
        
        Args:
            path: 웨이포인트 리스트
            current_waypoint_idx: 구조대원 현재 웨이포인트 인덱스
            penalty_field: 패널티 필드
            
        Returns:
            가중치 적용된 penalty_field
        """
        weighted_field = penalty_field.copy()
        
        if current_waypoint_idx >= len(path) - 1:
            return weighted_field
        
        # 향후 몇 개 웨이포인트에 할인 적용
        lookahead = min(self.config.direction_lookahead, len(path) - current_waypoint_idx)
        
        for i in range(current_waypoint_idx, current_waypoint_idx + lookahead):
            if i < len(path):
                row, col = path[i]
                if 0 <= row < weighted_field.shape[0] and 0 <= col < weighted_field.shape[1]:
                    # 15% 할인 = 0.85 배
                    weighted_field[int(row), int(col)] *= (1.0 - self.config.direction_weight_discount)
        
        return weighted_field
    
    def update_with_approval(self, 
                            approved: bool,
                            terrain: Dict,
                            wind_field: np.ndarray,
                            dem_lats: np.ndarray,
                            dem_lons: np.ndarray) -> Dict:
        """
        관제사 승인에 따른 경로 업데이트
        
        Args:
            approved: 승인 여부
            
        Returns:
            업데이트 결과 딕셔너리
        """
        result = {
            'success': False,
            'message': '',
            'new_path': None,
            'risk_reduction': 0.0
        }
        
        if not self.pending_new_path:
            result['message'] = "대기 중인 경로 변경이 없습니다."
            return result
        
        if approved:
            self.current_path = self.pending_new_path
            new_risk = self.evaluate_path_risk(self.current_path, terrain, wind_field,
                                               dem_lats, dem_lons)
            old_risk = self.current_risk_score
            risk_reduction = old_risk - new_risk
            
            self.current_risk_score = new_risk
            self.pending_new_path = None
            self.approval_pending = False
            
            result['success'] = True
            result['message'] = f"✅ 경로 변경 승인됨 (위험도 {risk_reduction*100:.1f}% 감소)"
            result['new_path'] = self.current_path
            result['risk_reduction'] = risk_reduction
        else:
            self.pending_new_path = None
            self.approval_pending = False
            result['message'] = "❌ 경로 변경이 거절되었습니다. 기존 경로 유지."
        
        return result
    
    def _get_max_wind_on_path(self, path: List[Tuple[int, int]], 
                             wind_field: np.ndarray) -> float:
        """경로상의 최대 풍속 반환"""
        max_wind = 0.0
        for row, col in path:
            if 0 <= row < wind_field.shape[0] and 0 <= col < wind_field.shape[1]:
                max_wind = max(max_wind, wind_field[int(row), int(col)])
        return max_wind
    
    def get_status_summary(self) -> Dict:
        """현재 상태 요약"""
        return {
            'wind_locked': self.wind_locked,
            'current_risk': self.current_risk_score,
            'approval_pending': self.approval_pending,
            'current_waypoint': self.current_waypoint_idx,
            'path_length': len(self.current_path) if self.current_path else 0
        }


class ApprovalManager:
    """경로 변경 승인 관리"""
    
    def __init__(self):
        self.pending_approval: Optional[Dict] = None
        self.approval_history: List[Dict] = []
    
    def request_approval(self, 
                        old_path: List[Tuple[int, int]],
                        new_path: List[Tuple[int, int]],
                        reason: str,
                        risk_reduction: float) -> Dict:
        """경로 변경 승인 요청"""
        request = {
            'timestamp': datetime.now(),
            'old_path_length': len(old_path),
            'new_path_length': len(new_path),
            'reason': reason,
            'risk_reduction': risk_reduction,
            'approved': None
        }
        self.pending_approval = request
        return request
    
    def respond_approval(self, approved: bool) -> Dict:
        """승인 응답"""
        if not self.pending_approval:
            return {'error': '대기 중인 승인 요청이 없습니다.'}
        
        self.pending_approval['approved'] = approved
        self.approval_history.append(self.pending_approval)
        result = self.pending_approval.copy()
        self.pending_approval = None
        
        return result
    
    def get_history(self, limit: int = 10) -> List[Dict]:
        """최근 승인 이력"""
        return self.approval_history[-limit:]
