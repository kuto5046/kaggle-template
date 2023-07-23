# kaggle-template
kaggleコンペ用のテンプレートレポジトリ

scriptやyamlを用意しているが基本的にはnotebookで完結することを目指しており、関数をコピペすればkaggle notebookでも実行可能なようにしている
(そのためhydraは使用していない。便利なので使いたいがnotebookでのいい運用が定まっていない)

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


## 実験の流れ
notebookでの実行を想定.1実験1notebook。
notebookフォルダにあるテンプレートをexpにcopyする。ファイル名はexp001.ipynbのような形式を想定


## その他メモ
- tqdmのループ内ではprintの代わりにtqdm.write()を使うと表示を崩さずprintできる
- loss:BCELoss, score: F1みたいなパターンだとBCELossのtargetはfloatでF1のtargetはlongである必要があるので注意 
- 分類は基本CEでマルチラベルとかの場合はBCEを使う