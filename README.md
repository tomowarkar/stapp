# stapp

`streamlit` と `opencv` を用いた画像処理アプリ

## Prerequisites

- [opencv](https://github.com/opencv/opencv)
- [streamlit](https://github.com/streamlit)

## Getting Started

以下の 2 行でローカル環境にて試すことができます。

```
$ pip install streamlit
$ streamlit run https://github.com/tomowarkar/stapp/blob/master/app.py
```

Cascade ページを利用する場合は分類器のダウンロードが必要です

[source](https://github.com/opencv/opencv/tree/master/data)

```
$ curl -sO https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
$ curl -sO https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml
```

## Sample Page

see [here](https://still-sands-82310.herokuapp.com/)

## Authors

[**tomowarkar**](https://github.com/tomowarkar)
