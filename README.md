# kaggle-template
kaggleコンペ用のテンプレートレポジトリ

## クイックスタート

### 前提条件
以下のツールがインストールされていることを前提としています。
- Docker & Docker Compose
- [mise](https://github.com/jdx/mise)

### セットアップ

1. GitHub上でUse this templateをクリックして新しいリポジトリを作成

2. プロジェクトのクローンとセットアップ
```bash
git clone <your-repo>
cd <your-repo>
```

3. 開発環境の起動
ホスト環境で以下を実行
```bash
mise run docker-build
```
4. VS CodeのRemote-Containers拡張機能でコンテナに接続

5. mise install
以下を実行することでuvやその他のツールがインストールされます。
```bash
mise install
```

## 開発ワークフロー

### Python環境の初期化

コンテナ内で以下を実行
```bash
mise run uv-setup
```

### 実験追跡

```bash
# Weights & Biasesにログイン
wandb login
```

### データセット管理

```bash
# 利用可能なデータセットの一覧表示
uv run kaggle datasets list

# コンペティションデータのダウンロード
cd input
kaggle datasets download <DATASET_NAME>

# 整理されたディレクトリに展開
unzip <DATASET_NAME>.zip -d <DATASET_NAME>
```

## プロジェクト構成

```
kaggle-template/
├── exp/                 # 実験ディレクトリ (exp001, exp002, ...)
├── input/               # コンペティションデータセット
├── notebook/            # EDA用Jupyterノートブック
├── output/              # モデル出力、特徴量、予測結果
├── src/                 # 再利用可能なユーティリティとツール
├── Dockerfile           # コンテナ設定
├── compose.yml          # Docker Compose設定
├── justfile            # タスク自動化
├── mise.yaml            # mise設定
```

## 実験管理

### 実験構成

実験フォルダの構成は以下。コンペに応じて自由に変更する。
ルール
- 1実験1ディレクトリ
- 実験ディレクトリは `exp/exp###` の形式で命名
- コードの実行は1runごとに分離する。runは同一の実験内で異なる設定やパラメータを試すための単位。
- 実装コードが大きく変わらない場合は同一exp内でrunを分けて実行する。

```bash
exp/exp001/
├── configs/             # runごとの設定ファイルを置く
│   ├── run0.yaml        # runごとの実験条件やパラメータを記述する
├── config.yaml          # 実験で共通するデフォルトの設定値を記述する
├── data_processor.py    # データの前処理や特徴量生成などtrain/inferenceの前に実施しておくと良い処理を行う
├── train.py            # 学習
└── inference.py        # 推論
```

### 設定管理

このテンプレートは[Hydra](https://hydra.cc/)を使用した設定管理を採用：

- **ベース設定**: `exp/*/config.yaml` - デフォルト実験設定
- **ラン設定**: `exp/*/configs/*.yaml` - 特定のラン用バリエーション
- **オーバーライド**: コマンドライン引数が優先

例：

```bash
# 学習率のオーバーライド
uv run python exp/exp001/train.py model.learning_rate=0.001

# 異なる設定ファイルの使用
uv run python exp/exp001/train.py --configs=run0
```

### 出力の管理

- `output/`: モデル、特徴量、予測結果を保存
- 実験ごとにサブディレクトリを作成して整理
- 重要なアーティファクトはW&Bにも記録
