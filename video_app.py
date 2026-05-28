import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium", app_title="PG Video Analysis")


@app.cell
def _():
    import marimo as mo
    import base64
    import os
    import tempfile
    import cv2

    return base64, cv2, mo, os, tempfile


@app.cell
def _():
    from dotenv import load_dotenv
    load_dotenv()
    from predictionguard import PredictionGuard
    client = PredictionGuard()
    return (client,)


@app.cell
def _(mo):
    mo.md("""
    # Video Frame Analysis with Prediction Guard's Vision Model
    **Model:** `Qwen3-VL-235B-A22B`

    Upload a short video clip. The app extracts evenly-spaced frames and sends them to the model together with your prompt.
    """)
    return


@app.cell
def _(mo):
    video_upload = mo.ui.file(
        filetypes=["video/mp4", "video/webm", "video/quicktime", "video/x-msvideo", "video/mpeg"],
        label="Upload Video (short clips recommended, < 50 MB)",
    )
    frame_slider = mo.ui.slider(
        start=4, stop=16, step=2, value=4,
        label="Frames to extract",
        show_value=True,
    )
    prompt = mo.ui.text_area(
        label="Prompt",
        value="Describe what is happening in this video. Walk through the sequence of events shown across the frames.",
        rows=3,
    )
    submit_btn = mo.ui.run_button(label="Analyze Video")
    return frame_slider, prompt, submit_btn, video_upload


@app.cell
def _(frame_slider, mo, prompt, submit_btn, video_upload):
    mo.vstack(
        [
            video_upload,
            mo.hstack([frame_slider, submit_btn], gap=2, align="end"),
            prompt,
        ],
        gap=2,
    )
    return


@app.cell
def _(base64, cv2, frame_slider, os, tempfile, video_upload):
    _files = video_upload.value
    frames_b64 = []

    if _files:
        _name = _files[0].name
        _suffix = ("." + _name.rsplit(".", 1)[-1]) if "." in _name else ".mp4"

        _tmp = tempfile.NamedTemporaryFile(suffix=_suffix, delete=False)
        try:
            _tmp.write(_files[0].contents)
            _tmp.close()

            _cap = cv2.VideoCapture(_tmp.name)
            _total = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            _n = frame_slider.value

            if _total > 0:
                for _i in range(_n):
                    _idx = int(_i * _total / _n)
                    _cap.set(cv2.CAP_PROP_POS_FRAMES, _idx)
                    _ret, _frame = _cap.read()
                    if _ret:
                        _, _buf = cv2.imencode(
                            ".jpg", _frame, [cv2.IMWRITE_JPEG_QUALITY, 80]
                        )
                        frames_b64.append(
                            base64.b64encode(_buf.tobytes()).decode()
                        )
            _cap.release()
        finally:
            os.unlink(_tmp.name)
    return (frames_b64,)


@app.cell
def _(frames_b64, mo):
    mo.stop(not frames_b64)
    _thumbs = [
        mo.image(src=f"data:image/jpeg;base64,{b}", width=160)
        for b in frames_b64
    ]
    _rows = [_thumbs[i : i + 4] for i in range(0, len(_thumbs), 4)]
    mo.vstack(
        [
            mo.md(f"**{len(frames_b64)} frames extracted:**"),
            *[mo.hstack(row, gap=1) for row in _rows],
        ],
        gap=1,
    )
    return


@app.cell
def _(client, frames_b64, mo, prompt, submit_btn):
    mo.stop(
        not submit_btn.value,
        mo.md("*Extract frames, enter a prompt, then click **Analyze Video**.*"),
    )
    mo.stop(
        not frames_b64,
        mo.callout(mo.md("**No video uploaded.** Please upload a video first."), kind="warn"),
    )

    _content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}
        for b in frames_b64
    ]
    _content.append({"type": "text", "text": prompt.value.strip() or "Describe what is happening in this video."})

    import concurrent.futures as _cf

    _output = None
    try:
        with mo.status.spinner(title=f"Analyzing {len(frames_b64)} frames with Qwen3-VL-235B-A22B…"):
            with _cf.ThreadPoolExecutor(max_workers=1) as _pool:
                _future = _pool.submit(
                    client.chat.completions.create,
                    model="Qwen3-VL-235B-A22B",
                    messages=[{"role": "user", "content": _content}],
                    max_completion_tokens=4096,
                    temperature=0.7,
                )
                response = _future.result()
        _answer = response["choices"][0]["message"]["content"]
        _output = mo.vstack(
            [
                mo.md("## Response"),
                mo.callout(mo.md(_answer), kind="success"),
            ],
            gap=2,
        )
    except Exception as e:
        _output = mo.callout(mo.md(f"**Error:** `{e}`"), kind="danger")
    mo.output.replace(_output)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
