import cudf
from utils import class_print, print_sys_usage

class DataLoader:
    def __init__(self):
        pass
    
    def load_raw_parquet(self, parquet_path: str, feature_cols: list) -> cudf.DataFrame:
        '''오리지널 parquet 파일 피쳐 중 feature cols에 있는 값들만 로드합니다.'''
        class_print(self, "raw parquet 파일을 로드합니다.")
        df = cudf.read_parquet(parquet_path, columns=feature_cols)
        class_print(self, f"raw parquet 로드 완료. shape={df.shape}")
        print_sys_usage("after load_raw_parquet")
        return df
    
    def load_parquet(self, parquet_path: str) -> cudf.DataFrame:
        '''전처리 후 저장된 parqeut 파일을 로드합니다.'''
        class_print(self, "processed parquet 파일을 로드합니다.")
        df = cudf.read_parquet(parquet_path)
        class_print(self, f"processed parquet 로드 완료. shape={df.shape}")
        print_sys_usage("after load_parquet")
        return df
    
    def save_parquet(self, df: cudf.DataFrame, save_path: str, comp: str = "snappy") -> None:
        '''전처리가 완료된 DataFrame을 parquet 파일로 저장합니다.'''
        class_print(self, "parquet 파일을 저장합니다.")
        df.to_parquet(save_path, compression=comp)
        class_print(self, f"저장 완료 → {save_path}")
        print_sys_usage("after save_parquet")
    