from dataclasses import dataclass

@dataclass
class RAFTConfig:
    # 기본 dropout 값 설정
    dropout: float = 0
    # 대체 상관 블록 사용 여부
    alternate_corr: bool = False
    # 작은 모델 사용 여부
    small: bool = False
    # 혼합 정밀도 사용 여부
    mixed_precision: bool = False

    def __iter__(self):
        # 설정된 모든 속성을 순회할 수 있도록 합니다.
        for k in self.__dict__:
            yield k
