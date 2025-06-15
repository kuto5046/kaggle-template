# kaggle-template
kaggleコンペ用のテンプレートレポジトリ

## 環境構築

### コンテナ起動

dockerで環境構築を行う。direnvを使用してホスト環境のUIDを自動的にコンテナに反映させる。

```bash
# direnvが未インストールの場合はインストール
# macOS: brew install direnv
# Ubuntu: sudo apt-get install direnv

# direnvを有効化（初回のみ、使用しているシェルに応じて実行）
# bash: echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
# zsh:  echo 'eval "$(direnv hook zsh)"' >> ~/.zshrc

# プロジェクトディレクトリでdirenvを許可
direnv allow

# 必要に応じて.envファイルを作成（プロジェクト固有の環境変数がある場合）
# cp .env.example .env

# docker composeでビルド・起動（direnvが自動的にUID等を設定）
docker compose up -d --build
```

あとはvscodeのdevcontainerでコンテナに入って作業する

### 仮想環境作成
uvを利用している。コンテナ起動時は仮想環境が作られていないためコンテナに入ったら以下を実行

```bash
uv sync
```

pre-commitでformatter, linter, type checkerを適用している  
以下を実行するとpre-commitが有効になる。必要に応じて`.pre-commit-config.yaml`を編集する

```bash
uv run pre-commit install
```

### wandb有効化

ターミナルで以下を実行
```bash
wandb login
```
authorizeすることでwandbが利用可能になる

### dataset準備
docker内だとkaggle APIが有効になっている

dataset一覧チェック

```bash
kaggle datasets list
```
datasetをdownload
```bash
cd input
kaggle datasets download <DATASET_NAME>
```
解凍(同じ名前のディレクトリを作成してその中に解凍)
```bash
unzip <DATASET_NAME>.zip -d <DATASET_NAME>
```

## ディレクトリ構成
主要なもののみ記載
```
root
├── exp/                 # 実験コードをここで管理する
├── input/               # コンペ用のデータセット置き場
├── notebook/            # EDAやdebug用のnotebookをおく
├── output/              # 特徴量やモデルなどの実験による出力を格納する 
├── src                  # 実験に依存しないコードをここに置く(kaggle datasetへのuploadなど)
```

## 実験方法
実験フォルダの構成は以下。コンペに応じて自由に変更する。
```
exp/exp001
|── configs              # runごとの設定ファイルをおく
├── config.yaml          # 実験で共通するデフォルトの設定値を記述する
├── data_processor.py    # データの前処理や特徴量生成などtrain/inferenceの前に実施しておくと良い処理を行う
├── inference.py         # 推論コード
├── train.py             # 学習コード
```

exp配下に新しい実験フォルダを作成して1実験1ディレクトリで実施する。  
templateではhydraを使っており、`config.yaml`にパラメータ管理をしている。  
1つの実験フォルダ内で複数のrunを行う場合は`configs`ディレクトリにrunごとの設定ファイルを作成する。
runを分ける例としては、foldを変えて実験する場合や、ハイパーパラメータを変えて実験する場合などがある。

```bash
uv run python exp/exp001/data_processor.py
uv run python exp/exp001/train.py
uv run python exp/exp001/inference.py
```

## コマンド実行
繰り返し行うようなコマンドは`justfile`に記載しタスク化している。  
例えばkaggle datasetへのcodeやmodelのアップロードはsrc/tools配下のpythonファイルを編集した上で以下を実行する。

```bash
just upload
```
