# Copyright 2020 tomowarkar

import cv2
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance


def main():
    """Page handler"""

    st.title("Tomowarkar")
    pages = ["App", "Cascade"]
    choice = st.sidebar.selectbox("Select Page", pages)
    st.subheader(choice)

    if choice == "App":
        im = load_image()
        app(im)

    if choice == "Cascade":
        im = load_image()
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
    except:
        return None


def load_image():
    default = "https://1.bp.blogspot.com/-nB77P4LkQC8/XWS5gdVF9xI/AAAAAAABUTM/2ilcEL7lWaICdqSRUpkxiAoxHMS9qqIQwCLcBGAs/s550/group_young_world.png"
    in_url = st.text_input("Image from URL", value=default)
    if in_url == default:
        st.warning(
            "Image from [いらすとや](https://www.irasutoya.com/2019/10/blog-post_448.html)"
        )
    in_img = st.file_uploader(
        "or Upload Image (This takes priority)", type=["jpg", "png", "jpeg"]
    )

    if in_img is None:
        im = image_from_url(in_url) if any(in_url) else None
    else:
        im = image_from_file(in_img)

    return im


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

        elif choice == "Hist":
            img = im
            arr = np.array(im.convert("RGB"))

            for i, col in enumerate(("b", "g", "r")):
                histr = cv2.calcHist([arr], [i], None, [256], [0, 256])
                plt.plot(histr, color=col)
                plt.xlim([0, 256])

            st.pyplot()

        else:
            img = im

        st.image(img, width=width)


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
    main()
