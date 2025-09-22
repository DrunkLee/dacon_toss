from dataloader import DataLoader
from preprocessor import Preprocessor
from train import Trainer
from utils import class_print

class Pipeline:
    def __init__(self, model_name:str, params:dict, target:str, n_splits:int=5, metric:str='auc'):
        self.dl = DataLoader()
        self.pp = Preprocessor()
        self.tr = Trainer(model_name, params, n_splits=n_splits, metric=metric)
        self.target = target
        
    def run_train(self, processed_parquet:str, ckpt_path:str, use_optuna:bool=True, n_trials:int=30):
        gdf = self.dl.load_processed_parquet(processed_parquet)
        df  = self.dl.to_pandas(gdf)

        X = self.pp.fit_transform(df, target=self.target)
        y = df[self.target].astype(int)

        if use_optuna:
            self.tr.tune(X, y, n_trials=n_trials)

        self.tr.fit_cv(X, y, refit_full=True)
        self.tr.save(ckpt_path)
        class_print(self, "학습 파이프라인 완료")

    def run_infer(self, processed_parquet:str, ckpt_path:str, out_csv:str, id_col:str|None=None):
        from inference import InferenceRunner
        gdf = self.dl.load_processed_parquet(processed_parquet)
        df  = self.dl.to_pandas(gdf)
        ir = InferenceRunner(ckpt_path)
        ir.predict_to_csv(df, out_csv, id_col=id_col)
        class_print(self, "추론 파이프라인 완료")