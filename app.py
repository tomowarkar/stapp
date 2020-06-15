# Copyright 2020 tomowarkar

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance


def main():
    """Page handler"""

    st.title("Tomowarkar")
    pages = ["About", "App", "Cascade"]
    choice = st.sidebar.selectbox("Select Page", pages)
    st.subheader(choice)

    if choice == "About":
        st.info("Click on the upper left menu to select a page.")
        st.write(ABOUT_TEXT)

    if choice == "App":
        im = load_image()
        with st.spinner("Generating ..."):
            app(im)

    if choice == "Cascade":
        im = load_image()
        with st.spinner("Generating ..."):
            cascade(im)

    if choice == "MosaicVideo":
        from mosaic_video import mosaic

        mosaic()

    text = """
    Author: [tomowarkar](https://tomowarkar.github.io/blog)

    Page Source: [github](https://github.com/tomowarkar/stapp)
    """
    st.info(text)


# Common
@st.cache
def image_from_file(img):
    im = Image.open(img)
    return im


@st.cache
def image_from_url(url):
    import requests
    from io import BytesIO

    try:
        response = requests.get(url)
        im = Image.open(BytesIO(response.content))
        return im
    except OSError:
        return None


def load_image():
    default = "https://1.bp.blogspot.com/-nB77P4LkQC8/XWS5gdVF9xI/AAAAAAABUTM/2ilcEL7lWaICdqSRUpkxiAoxHMS9qqIQwCLcBGAs/s550/group_young_world.png"
    in_url = st.text_input("Image from URL", value=default)
    in_img = st.file_uploader(
        "or Upload Image (This takes priority)", type=["jpg", "png", "jpeg"]
    )

    if in_img is None and in_url == default:
        st.warning(
            "Image from [いらすとや](https://www.irasutoya.com/2019/10/blog-post_448.html)"
        )

    im = None
    if any(in_url):
        im = image_from_url(in_url)
        if im is None:
            st.error("Invalid URL")

    if in_img is not None:
        im = image_from_file(in_img)

    return im


def signature(img):
    sign = st.sidebar.text_input("")
    if sign == "hoge":
        return img
    return cv2.putText(
        img,
        "tomowarkar",
        (30, img.shape[0] - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        thickness=1,
    )


# Page App
def app(im):
    """Page App"""
    width, choice = app_sidebar()
    if im is not None:
        if choice == "Bilevel":
            img = im.convert("1")

        elif choice == "Greyscale":
            img = im.convert("L")

        elif choice == "Contrast":
            factor = st.sidebar.slider(choice, 0.0, 3.0, 1.0)
            enhancer = ImageEnhance.Contrast(im)
            img = enhancer.enhance(factor)

        elif choice == "Brightness":
            factor = st.sidebar.slider(choice, 0.0, 3.0, 1.0)
            enhancer = ImageEnhance.Brightness(im)
            img = enhancer.enhance(factor)

        elif choice == "Colorpick":
            arr = np.array(im.convert("RGB"))
            hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV)

            a, b = st.sidebar.slider(choice, 0, 255, (0, 255))
            img_mask = cv2.inRange(hsv, np.ones(3) * a, np.ones(3) * b)
            img = cv2.bitwise_and(arr, arr, mask=img_mask)

        elif choice == "Canny":
            arr = np.array(im.convert("RGB"))
            gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            a, b = st.sidebar.slider(choice, 0, 500, (200, 400))
            img = cv2.Canny(gray, a, b)
            img = signature(img)

        elif choice == "Hist":
            from matplotlib import pyplot as plt

            arr = np.array(im.convert("RGB"))

            for i, col in enumerate(("b", "g", "r")):
                histr = cv2.calcHist([arr], [i], None, [256], [0, 256])
                plt.plot(histr, color=col)
                plt.xlim([0, 256])

            st.pyplot()

        elif choice == "Kuwahara":
            # https://en.wikipedia.org/wiki/Kuwahara_filter
            # https://qiita.com/Cartelet/items/5c1c012c132be3aa9608
            r = st.sidebar.slider(choice, 5, 50, 5, 5)
            arr = np.array(im.convert("RGB"))
            h, w, _ = arr.shape
            img = np.empty_like(arr)
            arr = np.pad(arr, ((r, r), (r, r), (0, 0)), "edge")
            ave, var = cv2.integral2(arr)
            ave_mask = (
                ave[: -r - 1, : -r - 1]
                + ave[r + 1 :, r + 1 :]
                - ave[r + 1 :, : -r - 1]
                - ave[: -r - 1, r + 1 :]
            )
            ave = ave_mask / (r + 1) ** 2

            var_mask = (
                var[: -r - 1, : -r - 1]
                + var[r + 1 :, r + 1 :]
                - var[r + 1 :, : -r - 1]
                - var[: -r - 1, r + 1 :]
            )
            var = (var_mask / (r + 1) ** 2 - ave ** 2).sum(axis=2)

            for i in range(h):
                for j in range(w):
                    a1, b1, c1, d1, = (
                        ave[i, j],
                        ave[i + r, j],
                        ave[i, j + r],
                        ave[i + r, j + r],
                    )
                    a2, b2, c2, d2, = (
                        var[i, j],
                        var[i + r, j],
                        var[i, j + r],
                        var[i + r, j + r],
                    )
                    img[i, j] = np.array([a1, b1, c1, d1])[
                        np.array([a2, b2, c2, d2]).argmin()
                    ]
            img = signature(img)

        try:
            st.image(img, width=width)
        except UnboundLocalError:
            st.image(im, width=width)


def app_sidebar():
    width = st.sidebar.slider("image width", 100, 1000, 600, 50)
    processing = [
        "Original",
        "Bilevel",
        "Greyscale",
        "Contrast",
        "Brightness",
        "Colorpick",
        "Canny",
        "Hist",
        "Kuwahara",
    ]
    choice = st.sidebar.radio("Processing", processing)
    return width, choice


# Page Cascade
def cascade(im):
    """Page Cascade"""
    width, choice, color, scaleFactor, minNeighbors = cascade_sidebar()

    if im is not None:
        arr = np.array(im.convert("RGB"))
        img = cv2.cvtColor(arr, 1)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

        if "Face" == choice:
            cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        if "Eye" == choice:
            cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

        rects = cascade.detectMultiScale(
            gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors
        )
        st.success(f"Found {len(rects)} parts")
        for (x, y, w, h) in rects:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=3)
        st.image(img, width=width)


def cascade_sidebar():
    width = st.sidebar.slider("image width", 100, 1000, 600, 50)

    parts = ["Face", "Eye"]
    choice = st.sidebar.radio("Face Parts", parts)

    c = st.sidebar.beta_color_picker("Color", "#ff0000").lstrip("#")
    color = tuple(int(c[i : i + 2], 16) for i in (0, 2, 4))

    scaleFactor = st.sidebar.slider("scaleFactor", 1.01, 2.0, 1.1)
    minNeighbors = st.sidebar.slider("minNeighbors", 0, 10, 4)

    return width, choice, color, scaleFactor, minNeighbors


if __name__ == "__main__":
    ABOUT_TEXT = """
        ## Welcome to this app!\n
        このウェブサイトは[tomowarkar](https://tomowarkar.github.io/blog)によって作成されたデモページです。\n
        Pythonを用いて開発され、Herokuによってホストされています。\n
        左側にあるページ選択からお好きなページへとお移りください。\n
        ## Pages info\n
        ### App\n
        [OpenCV](https://opencv.org/)を使った様々な画像加工を行う。\n
        ### Cascade\n
        [OpenCV](https://opencv.org/)の顔判定モデル使った人間の顔判定を行う。\n
        """

    main()
