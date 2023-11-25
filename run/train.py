from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.features.base import load_datasets
from src.models.gbdt import get_callbacks
from src.models.gbdt import get_model
from src.utils.visualize import plot_importance


class TrainPipeline:
    def __init__(self, cfg):
        # ここで利用するconfigは全て個別に初期化する
        self.cfg = cfg
        self.models = []
        self.scores = []
        self.rounds = []
        self.valid_preds = []
        self.valid_targets = []
        self.test_preds = []

    def setup_dataset(self, fold_idx=None):
        self.X_train = load_datasets(
            self.cfg.features, Path(self.cfg.path.feature_dir), phase="train", fold=fold_idx
        )
        self.X_valid = load_datasets(
            self.cfg.features, Path(self.cfg.path.feature_dir), phase="valid", fold=fold_idx
        )
        self.y_train = load_datasets(
            self.cfg.targets, Path(self.cfg.path.feature_dir), phase="train", fold=fold_idx
        )
        self.y_valid = load_datasets(
            self.cfg.targets, Path(self.cfg.path.feature_dir), phase="valid", fold=fold_idx
        )
        self.X_test = load_datasets(
            self.cfg.features, Path(self.cfg.path.feature_dir), phase="test", fold=fold_idx
        )

    def setup_callbacks(self):
        self.callbacks = get_callbacks(self.cfg.model.name)

    def setup_model(self):
        self.model = get_model(
            self.cfg.model.name,
            dict(self.cfg.model.params),
            self.cfg.model.num_boost_round,
            [],
            Path("./"),
            self.callbacks,
        )

    def train(self, fold_idx=None):
        self.model.train(self.X_train, self.y_train, self.X_valid, self.y_valid)
        self.rounds.append(self.model.model.best_iteration)
        self.model.save(fold_idx)
        self.models.append(self.model.model)

    def run(self):
        self.setup_callbacks()
        self.setup_model()
        for fold_idx in self.cfg.use_fold:
            self.setup_dataset(fold_idx)
            self.train(fold_idx)
            self.create_oof(fold_idx)
            self.inference(fold_idx)
            self.evaluate(fold_idx)
        print(f"total score = {np.mean(self.scores):.3f} +- {np.std(self.scores):.3f}")
        plot_importance(self.models, Path("./"))
        self.create_submission()

    def create_oof(self, fold_idx):
        pred = self.model.predict(self.X_valid)
        np.save(f"pred_valid_{fold_idx}", pred)
        self.valid_preds.append(pred)
        self.valid_targets.append(self.y_valid.to_numpy())

    def inference(self, fold_idx):
        pred = self.model.predict(self.X_test)
        np.save(f"pred_test_{fold_idx}", pred)
        self.test_preds.append(pred)

    def evaluate(self, fold_idx):
        score = calc_score(self.valid_targets[-1], self.valid_preds[-1], self.cfg.threshold)
        print(f"fold={fold_idx} score={score:.3f}")
        self.scores.append(score)

    def create_submission(self):
        test_df = pd.read_csv(Path(self.cfg.path.input_dir) / "test.csv")
        y_pred = np.mean(self.test_preds, axis=0)
        test_df["is_kokuhou"] = (y_pred > self.cfg.threshold).astype(int)
        test_df[["is_kokuhou"]].to_csv(Path(self.cfg.path.sub_dir) / "submission.csv", index=False)


def calc_score(target, pred, threshold):
    pred = (pred > threshold).astype(int)
    return f1_score(target.flatten(), pred)


@hydra.main(config_path="../conf", config_name="train", version_base="1.2")
def main(cfg):
    pipeline = TrainPipeline(cfg)
    pipeline.run()


if __name__ == "__main__":
    main()
