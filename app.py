import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium", app_title="PG Image Analysis")


@app.cell
def _():
    import marimo as mo
    import base64

    return base64, mo


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
    # Prediction Guard Vision Demo
    **Model:** `Qwen3-VL-235B-A22B`

    Upload an image and ask anything about it.
    """)
    return


@app.cell
def _(mo):
    image_upload = mo.ui.file(
        filetypes=["image/jpeg", "image/jpg", "image/png", "image/webp", "image/gif"],
        label="Upload Image",
    )
    prompt = mo.ui.text_area(
        label="Prompt",
        placeholder="What would you like to know about the image?",
        value="Describe this image in detail.",
        rows=4,
    )
    submit_btn = mo.ui.run_button(label="Analyze Image")
    return image_upload, prompt, submit_btn


@app.cell
def _(image_upload, mo, prompt, submit_btn):
    mo.vstack(
        [
            mo.hstack(
                [image_upload, mo.vstack([prompt, submit_btn], gap=1)],
                gap=2,
                align="start",
            ),
        ],
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(base64, image_upload, mo):
    _files = image_upload.value
    mo.stop(not _files)
    _info = _files[0]
    _b64 = base64.b64encode(_info.contents).decode("utf-8")
    _n = _info.name.lower()
    _mime = "image/png" if _n.endswith(".png") else "image/gif" if _n.endswith(".gif") else "image/webp" if _n.endswith(".webp") else "image/jpeg"
    mo.image(src=f"data:{_mime};base64,{_b64}", width=480)
    return


@app.cell
def _(base64, client, image_upload, mo, prompt, submit_btn):
    mo.stop(
        not submit_btn.value,
        mo.md("*Upload an image, enter a prompt, then click **Analyze Image**.*"),
    )


    files = image_upload.value
    mo.stop(
        not files,
        mo.callout(mo.md("**No image uploaded.** Please select an image file first."), kind="warn"),
    )

    file_info = files[0]
    img_b64 = base64.b64encode(file_info.contents).decode("utf-8")

    _name = file_info.name.lower()
    mime = "image/png" if _name.endswith(".png") else "image/gif" if _name.endswith(".gif") else "image/webp" if _name.endswith(".webp") else "image/jpeg"

    data_uri = f"data:{mime};base64,{img_b64}"
    prompt_text = prompt.value.strip() or "Describe this image in detail."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    _output = None
    try:
        with mo.status.spinner(title="Analyzing image with Qwen3-VL-235B-A22B…"):
            response = client.chat.completions.create(
                model="Qwen3-VL-235B-A22B",
                messages=messages,
                max_completion_tokens=4096,
                temperature=0.7,
            )
        answer = response["choices"][0]["message"]["content"]
        _output = mo.vstack(
            [
                mo.md("## Response"),
                # mo.image(src=data_uri, width=480),
                mo.callout(mo.md(answer), kind="success"),
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


if __name__ == "__main__":
    app.run()
