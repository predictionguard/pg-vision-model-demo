import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium", app_title="PG Video Semantic Search")


@app.cell
def _():
    import marimo as mo
    import base64
    import os
    import tempfile
    import cv2
    import numpy as np

    return base64, cv2, mo, np, os, tempfile


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
    # Video Semantic Search with BridgeTower Embeddings
    **Model:** `bridgetower-large-itm-mlm-itc`

    Upload a short video clip. The app extracts evenly-spaced frames and indexes them using
    BridgeTower's multimodal embeddings — then lets you retrieve specific moments using natural language.

    > **Tip:** Keep queries short (≤ 15 words). BridgeTower has a 100-token text context.
    """)
    return


@app.cell
def _(mo):
    video_upload = mo.ui.file(
        filetypes=["video/mp4", "video/webm", "video/quicktime", "video/x-msvideo", "video/mpeg"],
        label="Upload Video (e.g. b-roll-car-pedestrian.mp4)",
    )
    frame_slider = mo.ui.slider(
        start=4, stop=20, step=2, value=20,
        label="Frames to extract",
        show_value=True,
    )
    embed_btn = mo.ui.run_button(label="Extract & Embed Frames")
    return embed_btn, frame_slider, video_upload


@app.cell
def _(embed_btn, frame_slider, mo, video_upload):
    mo.vstack(
        [
            video_upload,
            mo.hstack([frame_slider, embed_btn], gap=2, align="end"),
        ],
        gap=2,
    )
    return


@app.cell
def _(base64, cv2, frame_slider, os, tempfile, video_upload):
    _files = video_upload.value
    frames_b64 = []
    timestamps = []

    if _files:
        _name = _files[0].name
        _suffix = ("." + _name.rsplit(".", 1)[-1]) if "." in _name else ".mp4"
        _tmp = tempfile.NamedTemporaryFile(suffix=_suffix, delete=False)
        try:
            _tmp.write(_files[0].contents)
            _tmp.close()
            _cap = cv2.VideoCapture(_tmp.name)
            _total = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            _fps = _cap.get(cv2.CAP_PROP_FPS) or 30.0
            _n = frame_slider.value
            if _total > 0:
                for _i in range(_n):
                    _idx = int(_i * _total / _n)
                    _cap.set(cv2.CAP_PROP_POS_FRAMES, _idx)
                    _ret, _frame = _cap.read()
                    if _ret:
                        _, _buf = cv2.imencode(".jpg", _frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        frames_b64.append(base64.b64encode(_buf.tobytes()).decode())
                        timestamps.append(_idx / _fps)
            _cap.release()
        finally:
            os.unlink(_tmp.name)
    return frames_b64, timestamps


@app.cell
def _(frames_b64, mo, timestamps):
    mo.stop(not frames_b64)
    _thumbs = [
        mo.vstack(
            [
                mo.image(src=f"data:image/jpeg;base64,{b}", width=150),
                mo.md(f"<center>{t:.1f}s</center>"),
            ],
            gap=0,
        )
        for b, t in zip(frames_b64, timestamps)
    ]
    _rows = [_thumbs[i : i + 4] for i in range(0, len(_thumbs), 4)]
    mo.vstack(
        [
            mo.md(f"**{len(frames_b64)} frames ready — click Extract & Embed to index them:**"),
            *[mo.hstack(row, gap=1) for row in _rows],
        ],
        gap=1,
    )
    return


@app.cell
def _(client, embed_btn, frames_b64, mo):
    mo.stop(
        not embed_btn.value,
        mo.md("*Upload a video, adjust the frame count, then click **Extract & Embed Frames**.*"),
    )
    mo.stop(
        not frames_b64,
        mo.callout(mo.md("**No frames available.** Upload a video first."), kind="warn"),
    )

    frame_embeddings = None
    try:
        with mo.status.spinner(title=f"Embedding {len(frames_b64)} frames with BridgeTower…"):
            _resp = client.embeddings.create(
                model="bridgetower-large-itm-mlm-itc",
                input=[{"image": b} for b in frames_b64],
            )
            frame_embeddings = [item["embedding"] for item in _resp["data"]]
        mo.output.replace(
            mo.callout(
                mo.md(
                    f"**{len(frame_embeddings)} frames indexed.** "
                    f"Embedding dim: `{len(frame_embeddings[0])}` · Search below."
                ),
                kind="success",
            )
        )
    except Exception as _e:
        mo.output.replace(mo.callout(mo.md(f"**Embedding error:** `{_e}`"), kind="danger"))
    return (frame_embeddings,)


@app.cell
def _(mo):
    search_query = mo.ui.text(
        label="Search query",
        placeholder="e.g.'bicycle'",
        full_width=True,
    )
    top_k_slider = mo.ui.slider(
        start=1,
        stop=6,
        value=2,
        label="Top results",
        show_value=True,
    )
    search_btn = mo.ui.run_button(label="Search")
    return search_btn, search_query, top_k_slider


@app.cell
def _(mo, search_btn, search_query, top_k_slider):
    mo.vstack(
        [
            mo.md("## Search Video Frames"),
            mo.hstack([search_query, top_k_slider, search_btn], gap=2, align="end"),
        ],
        gap=1,
    )
    return


@app.cell
def _(
    client,
    frame_embeddings,
    frames_b64,
    mo,
    np,
    search_btn,
    search_query,
    timestamps,
    top_k_slider,
):
    mo.stop(
        not search_btn.value,
        mo.md("*Enter a query and click **Search** to find matching frames.*"),
    )
    mo.stop(
        not frame_embeddings,
        mo.callout(
            mo.md("**Frames not indexed yet.** Click **Extract & Embed Frames** above first."),
            kind="warn",
        ),
    )
    mo.stop(
        not search_query.value.strip(),
        mo.callout(mo.md("**Empty query.** Please enter a search term."), kind="warn"),
    )

    try:
        with mo.status.spinner(title="Embedding query and ranking frames…"):
            _q_resp = client.embeddings.create(
                model="bridgetower-large-itm-mlm-itc",
                input=[{"text": search_query.value.strip()}],
            )
        _q_emb = np.array(_q_resp["data"][0]["embedding"])
        _f_embs = np.array(frame_embeddings)
        _norms = np.linalg.norm(_f_embs, axis=1) * np.linalg.norm(_q_emb)
        _sims = (_f_embs @ _q_emb) / np.maximum(_norms, 1e-8)

        _k = min(top_k_slider.value, len(frames_b64))
        _top_idx = np.argsort(_sims)[::-1][:_k]

        _result_cards = [
            mo.vstack(
                [
                    mo.image(src=f"data:image/jpeg;base64,{frames_b64[_i]}", width=200),
                    mo.md(
                        f"**#{_rank + 1}** · {timestamps[_i]:.1f}s\n\n"
                        f"Score: `{float(_sims[_i]):.4f}`"
                    ),
                ],
                gap=0,
            )
            for _rank, _i in enumerate(_top_idx)
        ]

        mo.output.replace(
            mo.vstack(
                [
                    mo.md(f'## Results for: *"{search_query.value.strip()}"*'),
                    mo.hstack(_result_cards, gap=2, wrap=True),
                ],
                gap=2,
            )
        )
    except Exception as _e:
        mo.output.replace(mo.callout(mo.md(f"**Search error:** `{_e}`"), kind="danger"))
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
