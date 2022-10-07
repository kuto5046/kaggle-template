# tabular-template
テーブルコンペのテンプレートレポジトリ

## 環境構築
dockerで環境構築を行う。
以下のレポジトリに従ってコンテナを作成する。
https://github.com/kuto5046/docker


## dataset準備
docker内だとkaggle APIが有効になっている

dataset一覧チェック
```
kaggle datasets list
```

datasetをdownload
```
cd input
kaggle datasets download <DATASET_NAME>
```
解凍(同じ名前のディレクトリを作成してその中に解凍)
```
unzip <DATASET_NAME>.zip -d <DATASET_NAME>
```

## 初めにすること

### ログ系
wandbのprojectをwebから作成
ターミナルで以下を実行
```sh
wandb login
```
authorizeすることでwandbが利用可能になる

### 通知系
以下のURLからwebhook URLをコピー
https://api.slack.com/apps/A014TUBTH8X/incoming-webhooks?

ルートディレクトリに.envファイルを作成し以下のように記載
```
SLACK_WEBHOOK_URL=<ここにペースト>
```



## 実験の流れ
基本的に全実験をscriptで行う
まずfeature_engineering.pyを実行し特徴量生成を行う。作成された特徴量はfeature_storeに保存される。

```sh
python feature_engineering.py
```

config/experiment/にdefault.yamlとの差分を作成しコマンドライン引数として渡す
```sh
python train.py experiment=example
```


どうしても現状のコードを残してスッキリ始めたいとなった場合はexp内のフォルダバージョンを繰り上げる。


