
import cv2
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
from PIL import Image, ImageEnhance


def main():
    """Page handler"""

    st.title("My App")

    pages = ["App", "Cascade"]

    choice = st.sidebar.selectbox("Select Page", pages)
    st.subheader(choice)

    if choice == "App":
        app()
    if choice == "Cascade":
        cascade()


# Common
@st.cache
def loaf_image(img):
    im = Image.open(img)
    return im


# Page App
def app():
    """Page App"""
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

    img_in = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if img_in is not None:
        im = loaf_image(img_in)
        st.text(choice)

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

            for i, col in enumerate(('b', 'g', 'r')):
                histr = cv2.calcHist([arr], [i], None, [256], [0, 256])
                plt.plot(histr, color=col)
                plt.xlim([0, 256])

            st.pyplot()

        else:
            img = im

        st.image(img, width=width)


# Page Cascade
def cascade():
    """Page Cascade"""
    width = st.sidebar.slider("image width", 100, 1000, 600, 50)

    parts = ["Face", "Eye"]
    choice = st.sidebar.radio("Face Parts", parts)

    c = st.sidebar.beta_color_picker("Color", "#ff0000").lstrip("#")
    color = tuple(int(c[i: i + 2], 16) for i in (0, 2, 4))

    scaleFactor = st.sidebar.slider("scaleFactor", 1.01, 2.0, 1.1)
    minNeighbors = st.sidebar.slider("minNeighbors", 0, 10, 4)

    img_in = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if img_in is not None:
        im = loaf_image(img_in)
        arr = np.array(im.convert("RGB"))
        img = cv2.cvtColor(arr, 1)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

        if "Face" == choice:
            cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        if "Eye" == choice:
            cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

        rects = cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        st.success(f"Found {len(rects)} parts")
        for (x, y, w, h) in rects:
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness=3)
        st.image(img, width=width)


if __name__ == "__main__":
    main()
