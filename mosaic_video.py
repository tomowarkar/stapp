import os
import tempfile
import uuid

import cv2
import numpy as np
import streamlit as st


@st.cache(suppress_st_warning=True)
def generate_video(text: str) -> bytes:
    scale = tuple(map(int, text.split(" x ")))
    filename = f"{str(uuid.uuid4())}.mp4"

    with tempfile.TemporaryDirectory() as td:
        filename = os.path.join(td, filename)
        st.write(filename)

        gen_video(filename, scale)
        video_file = open(filename, "rb")

    return video_file.read()


def gen_video(filename, scale):
    height, width, fps = 400, 600, 10
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    video = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    steps = fps * 10
    my_bar = st.progress(0)
    for i in range(steps):
        img = np.random.randint(0, 256, (scale[0], scale[1], 3), np.uint8)
        img = cv2.resize(img, (width, height))
        video.write(img)

        my_bar.progress((100 * i) // steps)

    video.release()


def mosaic():
    size = st.radio("pixel", ("2 x 3", "30 x 45", "400 x 600"))

    video = generate_video(size)
    st.video(video)
