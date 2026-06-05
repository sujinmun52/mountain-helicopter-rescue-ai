"""
실시간 동적 경로 최적화 시뮬레이션
- 시간에 따른 풍속 변화
- 경로 변경 필요성 판단
- 관제 승인 프로세스 시연
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Dict
from dataclasses import dataclass


# --- 동적 경로 최적화 클래스 임베드 ---

@dataclass
class PathUpdateConfig:
    """경로 업데이트 설정"""
    wind_threshold_m_s: float = 12.0
    wind_threshold_band: float = 1.0
    risk_increase_threshold: float = 0.25
    direction_weight_discount: float = 0.15
    direction_lookahead: int = 3
    min_update_interval_minutes: int = 10
    soft_update_enabled: bool = True
    require_approval: bool = True


class RealTimePathOptimizer:
    """실시간 경로 최적화 엔진"""
    
    def __init__(self, config: PathUpdateConfig = None):
        self.config = config or PathUpdateConfig()
        self.current_path = None
        self.current_risk_score: float = 0.0
        self.last_update_time = None
        self.current_waypoint_idx: int = 0
        self.approval_pending: bool = False
        self.pending_new_path = None
        self.wind_locked: bool = False
        
    def evaluate_path_risk(self, path, terrain, wind_field, dem_lats, dem_lons) -> float:
        if not path or len(path) < 2:
            return 0.0
        
        total_risk = 0.0
        segment_count = len(path) - 1
        
        for i in range(segment_count):
            row, col = path[i]
            if not (0 <= row < wind_field.shape[0] and 0 <= col < wind_field.shape[1]):
                continue
            
            wind_speed = wind_field[int(row), int(col)]
            wind_risk = min(wind_speed / 20.0, 1.0)
            
            slope = terrain.get('slope', np.zeros_like(wind_field))[int(row), int(col)]
            slope_risk = min(slope / 90.0, 1.0)
            
            is_open = terrain.get('is_open', np.ones_like(wind_field))[int(row), int(col)]
            forest_risk = 0.0 if is_open else 0.3
            
            is_ridge = terrain.get('is_ridge', np.zeros_like(wind_field))[int(row), int(col)]
            ridge_risk = 0.2 if is_ridge else 0.0
            
            segment_risk = (wind_risk * 0.4 + slope_risk * 0.3 + 
                           forest_risk * 0.2 + ridge_risk * 0.1)
            total_risk += segment_risk
        
        return total_risk / segment_count if segment_count > 0 else 0.0
    
    def should_update_path(self, current_path, new_path, wind_field, terrain, dem_lats, dem_lons) -> Tuple[bool, str]:
        current_max_wind = self._get_max_wind_on_path(current_path, wind_field)
        threshold = self.config.wind_threshold_m_s
        
        if self.wind_locked:
            threshold += self.config.wind_threshold_band
        
        if current_max_wind <= threshold:
            self.wind_locked = True
            return False, f"풍속 {current_max_wind:.1f}m/s ≤ {threshold:.1f}m/s (경로 잠금)"
        
        self.wind_locked = False
        
        current_risk = self.evaluate_path_risk(current_path, terrain, wind_field, dem_lats, dem_lons)
        new_risk = self.evaluate_path_risk(new_path, terrain, wind_field, dem_lats, dem_lons)
        
        risk_increase_rate = (current_risk - new_risk) / current_risk if current_risk > 0 else 0
        
        if risk_increase_rate >= self.config.risk_increase_threshold:
            return True, f"위험도 {risk_increase_rate*100:.1f}% 감소 (임계치: 25%)"
        
        return False, f"위험도 감소 {risk_increase_rate*100:.1f}% < 25% (변경 안함)"
    
    def _get_max_wind_on_path(self, path, wind_field) -> float:
        max_wind = 0.0
        for row, col in path:
            if 0 <= row < wind_field.shape[0] and 0 <= col < wind_field.shape[1]:
                max_wind = max(max_wind, wind_field[int(row), int(col)])
        return max_wind
    
    def get_status_summary(self) -> Dict:
        return {
            'wind_locked': self.wind_locked,
            'current_risk': self.current_risk_score,
            'approval_pending': self.approval_pending,
            'current_waypoint': self.current_waypoint_idx,
            'path_length': len(self.current_path) if self.current_path else 0
        }
    
    def update_with_approval(self, approved: bool, terrain, wind_field, dem_lats, dem_lons) -> Dict:
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
            new_risk = self.evaluate_path_risk(self.current_path, terrain, wind_field, dem_lats, dem_lons)
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


class ApprovalManager:
    """경로 변경 승인 관리"""
    
    def __init__(self):
        self.pending_approval: Dict = None
        self.approval_history: List[Dict] = []
    
    def request_approval(self, old_path, new_path, reason: str, risk_reduction: float) -> Dict:
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
        if not self.pending_approval:
            return {'error': '대기 중인 승인 요청이 없습니다.'}
        
        self.pending_approval['approved'] = approved
        self.approval_history.append(self.pending_approval)
        result = self.pending_approval.copy()
        self.pending_approval = None
        
        return result
    
    def get_history(self, limit: int = 10) -> List[Dict]:
        return self.approval_history[-limit:]


def simulate_wind_changes(base_wind: float = 8.0, 
                         time_steps: int = 10) -> np.ndarray:
    """
    시간 경과에 따른 풍속 변화 시뮬레이션
    
    Args:
        base_wind: 기본 풍속 (m/s)
        time_steps: 시간 스텝 개수
        
    Returns:
        [time_steps, 200, 200] 풍속 필드
    """
    wind_history = np.zeros((time_steps, 200, 200))
    
    for t in range(time_steps):
        # 시간에 따른 풍속 변화
        # t=0: 8m/s, t=3: 14m/s (피크), t=6: 16m/s (경고)
        if t < 3:
            wind_speed = base_wind + (t * 2.0)  # 증가
        elif t < 5:
            wind_speed = base_wind + 6.0 + ((t - 3) * 1.0)  # 가파른 증가
        else:
            wind_speed = base_wind + 8.0 - ((t - 5) * 0.5)  # 천천히 감소
        
        # 공간적 변화 추가 (능선에서 더 강함)
        x = np.linspace(0, 10, 200)
        y = np.linspace(0, 10, 200)
        X, Y = np.meshgrid(x, y)
        
        # 능선 근처 (100~120 row) 에서 풍속 증가
        spatial_factor = 1.0 + 0.5 * np.exp(-((X - 5)**2 + (Y - 5.5)**2) / 2)
        
        wind_history[t] = wind_speed * spatial_factor + np.random.normal(0, 0.5, (200, 200))
        wind_history[t] = np.clip(wind_history[t], base_wind - 2, base_wind + 12)
    
    return wind_history


def create_dummy_terrain() -> Dict:
    """더미 지형 데이터 생성"""
    terrain = {
        'slope': np.random.uniform(15, 60, (200, 200)),
        'is_open': np.random.choice([True, False], (200, 200), p=[0.6, 0.4]),
        'is_ridge': np.random.choice([True, False], (200, 200), p=[0.15, 0.85]),
        'forest_map': np.random.uniform(0, 100, (200, 200))
    }
    return terrain


def create_dummy_path() -> List[Tuple[int, int]]:
    """더미 경로 생성 (119센터 → 착륙지점 → 조난자)"""
    path = []
    
    # 119센터 (200, 200) → 착륙지점 (100, 100)
    for i in range(50):
        t = i / 50.0
        row = int(200 - t * 100)
        col = int(200 - t * 100)
        path.append((row, col))
    
    # 착륙지점 (100, 100) → 조난자 (50, 80)
    for i in range(1, 20):
        t = i / 20.0
        row = int(100 - t * 50)
        col = int(100 - t * 20)
        path.append((row, col))
    
    return path


def create_alternative_path() -> List[Tuple[int, int]]:
    """더미 대체 경로 생성 (풍속이 낮은 경로)"""
    path = []
    
    # 우회 경로 (동쪽으로 편향)
    for i in range(50):
        t = i / 50.0
        row = int(200 - t * 100)
        col = int(200 - t * 100 + t * 30)  # 동쪽으로 편향
        path.append((row, col))
    
    for i in range(1, 20):
        t = i / 20.0
        row = int(100 - t * 50)
        col = int(130 - t * 50)
        path.append((row, col))
    
    return path


def print_simulation_step(step: int, 
                         time: datetime,
                         wind_field: np.ndarray,
                         optimizer: RealTimePathOptimizer,
                         approval_manager: ApprovalManager):
    """시뮬레이션 스텝 출력"""
    
    max_wind = np.max(wind_field)
    avg_wind = np.mean(wind_field)
    
    print(f"\n{'='*70}")
    print(f"[시간 {step}] {time.strftime('%H:%M')} - 기상 업데이트")
    print(f"{'='*70}")
    print(f"🌪️  현재 풍속: 최대 {max_wind:.1f}m/s, 평균 {avg_wind:.1f}m/s")
    print(f"📍 경로상 풍속: {wind_field[100, 100]:.1f}m/s")
    
    status = optimizer.get_status_summary()
    print(f"\n⚙️  시스템 상태:")
    print(f"   • 풍속 잠금: {'🔒 YES' if status['wind_locked'] else '🔓 NO'}")
    print(f"   • 현재 위험도: {status['current_risk']:.2f}")
    print(f"   • 경로 포인트: {status['path_length']}")
    
    if approval_manager.pending_approval:
        print(f"\n⚠️  대기 중인 경로 변경:")
        req = approval_manager.pending_approval
        print(f"   • 사유: {req['reason']}")
        print(f"   • 위험도 감소: {req['risk_reduction']*100:.1f}%")


def run_dynamic_pathfinding_simulation():
    """동적 경로 최적화 시뮬레이션 실행"""
    
    print("\n" + "="*70)
    print("🚁 실시간 동적 경로 최적화 시뮬레이션")
    print("="*70)
    print("\n시나리오:")
    print("1. 초기 풍속: 8m/s (안전)")
    print("2. 점진적 풍속 증가 → 14m/s (주의) → 16m/s (위험)")
    print("3. 풍속 감소 후 안정화")
    print("\n경로 변경 규칙:")
    print(f"  • 풍속 임계치: 12m/s (이상 시 경로 변경 검토)")
    print(f"  • 위험도 임계치: 25% (이상 감소 시 변경)")
    print(f"  • 방향 가중치: -15% (진행 방향 선호)")
    print("="*70)
    
    # 설정
    config = PathUpdateConfig(
        wind_threshold_m_s=12.0,
        risk_increase_threshold=0.25,
        direction_weight_discount=0.15
    )
    
    optimizer = RealTimePathOptimizer(config)
    approval_manager = ApprovalManager()
    terrain = create_dummy_terrain()
    dem_lats = np.linspace(38.0, 38.25, 200)
    dem_lons = np.linspace(128.25, 128.5, 200)
    dem_array = np.random.uniform(1000, 1800, (200, 200))
    
    # 초기 경로
    current_path = create_dummy_path()
    optimizer.current_path = current_path
    optimizer.current_risk_score = optimizer.evaluate_path_risk(
        current_path, terrain, np.full((200, 200), 8.0), dem_lats, dem_lons
    )
    
    # 풍속 시간 변화
    wind_history = simulate_wind_changes(base_wind=8.0, time_steps=10)
    
    start_time = datetime.now()
    
    for step in range(len(wind_history)):
        current_time = start_time + timedelta(minutes=step*5)
        wind_field = wind_history[step]
        
        # 상태 출력
        print_simulation_step(step, current_time, wind_field, optimizer, approval_manager)
        
        # 경로 변경 필요성 검토
        alternative_path = create_alternative_path()
        should_update, reason = optimizer.should_update_path(
            current_path, alternative_path, wind_field, terrain, dem_lats, dem_lons
        )
        
        print(f"\n🔍 경로 변경 검토:")
        print(f"   {reason}")
        
        if should_update:
            print(f"\n⚠️  경로 변경 제안됨!")
            print(f"   새 경로 예상 위험도: {optimizer.evaluate_path_risk(alternative_path, terrain, wind_field, dem_lats, dem_lons):.2f}")
            
            # 관제 승인 요청
            if config.require_approval:
                print(f"\n📞 관제사에게 승인 요청 중...")
                approval_manager.request_approval(
                    current_path, 
                    alternative_path,
                    reason,
                    optimizer.evaluate_path_risk(current_path, terrain, wind_field, dem_lats, dem_lons) - 
                    optimizer.evaluate_path_risk(alternative_path, terrain, wind_field, dem_lats, dem_lons)
                )
                
                # 시뮬레이션: 85% 확률로 승인
                import random
                approved = random.random() < 0.85
                
                if approved:
                    print(f"   ✅ 관제사가 경로 변경을 승인했습니다!")
                    result = optimizer.update_with_approval(True, terrain, wind_field, dem_lats, dem_lons)
                    current_path = alternative_path
                    optimizer.current_path = current_path
                else:
                    print(f"   ❌ 관제사가 경로 변경을 거절했습니다. (기존 경로 유지)")
                    optimizer.update_with_approval(False, terrain, wind_field, dem_lats, dem_lons)
        else:
            print(f"   ✅ 기존 경로 유지")
        
        # 진행 방향 가중치 적용 (3분마다 한 지점 이동)
        if step > 0 and step % 1 == 0:
            optimizer.current_waypoint_idx = min(optimizer.current_waypoint_idx + 3, len(current_path) - 1)
            print(f"\n   구조대원 진행: 웨이포인트 {optimizer.current_waypoint_idx}/{len(current_path)}")
        
        print(f"\n   대기 시간: 5분 (다음 기상 업데이트)")
    
    print(f"\n\n{'='*70}")
    print("📊 시뮬레이션 결과")
    print(f"{'='*70}")
    
    history = approval_manager.get_history()
    print(f"\n총 경로 변경 요청: {len(history)}")
    
    if history:
        approved_count = sum(1 for h in history if h['approved'])
        print(f"승인된 변경: {approved_count}/{len(history)} ({approved_count*100/len(history):.0f}%)")
        
        print(f"\n변경 이력:")
        for i, h in enumerate(history, 1):
            status = "✅ 승인" if h['approved'] else "❌ 거절"
            print(f"  {i}. {status} - 위험도 {h['risk_reduction']*100:.1f}% 감소 예상")
    
    print(f"\n✅ 시뮬레이션 완료!")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    run_dynamic_pathfinding_simulation()
