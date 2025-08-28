# ui.py ¡ª MedGemma-only front-end (matches your backend)
import os
import io
import requests
import gradio as gr

API_BASE = os.getenv("API_BASE", "http://localhost:5000")

# -----------------------
# Helpers talking to backend
# -----------------------
def get_model_list():
    """
    GET /api/models -> [{name, alias, infer_fn}, ...]
    Returns (choices, raw_list)
    """
    try:
        res = requests.get(f"{API_BASE}/api/models", timeout=10)
        res.raise_for_status()
        data = res.json()
        # dropdown choices: (label, value)
        choices = [
            (f"{item.get('alias', item['name'])} [{item.get('infer_fn','?')}]", item["name"])
            for item in data
        ]
        return choices, data
    except Exception as e:
        print("get_model_list error:", e)
        return [], []

def _infer_fn_for(model_name, models_raw):
    for item in (models_raw or []):
        if item.get("name") == model_name:
            return item.get("infer_fn", "?")
    return "?"

def _status_for(model_name, models_raw):
    if not model_name:
        return "No model selected"
    infer_fn = _infer_fn_for(model_name, models_raw)
    return f"Selected: {model_name} (adapter={infer_fn})"

def download_model(hf_or_local):
    if not hf_or_local:
        raise gr.Error("Please enter the Hugging Face model ID or local path")

    # Correct detection: only treat as local if it's an existing directory
    if os.path.isdir(hf_or_local):
        payload = {"local_model_dir": hf_or_local}
    else:
        payload = {"hf_model_id": hf_or_local}

    try:
        res = requests.post(f"{API_BASE}/api/download_model", json=payload, timeout=1800)
        res.raise_for_status()
        data = res.json()
    except Exception as e:
        raise gr.Error(f"Download request failed: {e}")

    if not data.get("success"):
        raise gr.Error(f"Model download failed: {data.get('error')}")

    choices, raw = get_model_list()
    first_model = (data.get("model_names") or [None])[0]
    return gr.update(choices=choices, value=first_model, interactive=True, info=""), raw


def refresh_models():
    """
    Manually re-pull /api/models and refresh dropdown, keeping current selection if still valid.
    """
    choices, raw = get_model_list()
    # Try to keep current value; Gradio passes component values only via .value on runtime objects,
    # but here we just pick the first one and let the caller chain a status update.
    new_value = choices[0][1] if choices else None
    return gr.update(choices=choices, value=new_value, interactive=bool(choices)), raw

def generate_report(image, user_prompt, model_name):
    """
    POST /api/upload with image + form fields {prompt, model_name}
    """
    if not model_name:
        raise gr.Error("Please select a model first")
    if image is None:
        raise gr.Error("Please upload an image")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    files = {'image': ('image.png', buf, 'image/png')}
    data = {'prompt': user_prompt or "", 'model_name': model_name}

    try:
        r = requests.post(f"{API_BASE}/api/upload", files=files, data=data, timeout=1800)
        out = r.json()
    except Exception as e:
        return f"Error: {e}"

    return out.get('report') if out.get('success') else f"Error: {out.get('error')}"

# -----------------------
# Gradio UI
# -----------------------
with gr.Blocks(title="Local Medical VLM (MedGemma)") as demo:
    # Initial fetch
    choices, raw_models = get_model_list()
    models_state = gr.State(raw_models)

    with gr.Row():
        model_list = gr.Dropdown(
            choices=choices,
            label="Select Model Variant",
            value=(choices[0][1] if choices else None),
            interactive=bool(choices),
            info="Download/register a model if list is empty",
            scale=3
        )
        search_box = gr.Textbox(label="HF model ID or local path", placeholder="e.g., google/medgemma-4b-it or /path/to/model", scale=3)
        with gr.Column(scale=2):
            download_btn = gr.Button("Download / Register", variant="primary")
            refresh_btn = gr.Button("Refresh list")

    status = gr.Markdown(_status_for(model_list.value if choices else None, raw_models))

    img = gr.Image(type="pil", label="Medical Image (PNG/JPG/TIF)", visible=True)
    prompt = gr.Textbox(lines=2, label="Doctor Prompt", placeholder="e.g., Describe key findings / impressions")
    output = gr.Textbox(label="Generated Report", lines=12)
    submit = gr.Button("Analyze")

    # Events
    download_btn.click(
        download_model,
        inputs=[search_box],
        outputs=[model_list, models_state],
    ).then(
        lambda m, s: _status_for(m, s),
        inputs=[model_list, models_state],
        outputs=[status]
    )

    refresh_btn.click(
        refresh_models,
        outputs=[model_list, models_state],
    ).then(
        lambda m, s: _status_for(m, s),
        inputs=[model_list, models_state],
        outputs=[status]
    )

    model_list.change(
        lambda m, s: _status_for(m, s),
        inputs=[model_list, models_state],
        outputs=[status]
    )

    submit.click(
        generate_report,
        inputs=[img, prompt, model_list],
        outputs=[output]
    )

    # On load, ensure status is set correctly even when list is empty
    demo.load(
        lambda m, s: _status_for(m, s),
        inputs=[model_list, models_state],
        outputs=[status]
    )

demo.launch(server_name='0.0.0.0', server_port=7860)
