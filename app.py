import gradio as gr
from ultralytics import YOLO
from PIL import Image, ImageDraw

# ---------------- MODEL ----------------
model = YOLO("best.pt")
CONF = 0.3


# ---------------- DETECTION ----------------
def detect(image):
    if image is None:
        return None, "NO SIGNAL", "0"

    img = image.convert("RGB")
    results = model.predict(img, conf=CONF)

    draw = ImageDraw.Draw(img)
    logs = []
    count = 0

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls].lower()
            conf = float(box.conf[0])

            if label not in ["person", "human"]:
                continue

            count += 1
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            draw.rectangle([x1, y1, x2, y2], outline="#00fff7", width=3)
            draw.text((x1, y1 - 14), f"HUMAN {conf:.2f}", fill="#00fff7")

            logs.append(f"> TARGET LOCKED | CONF={conf:.2f}")

    if count == 0:
        logs.append("> AREA CLEAR")

    return img, "\n".join(logs), str(count)


# ---------------- UI ----------------
with gr.Blocks() as demo:

    gr.HTML("""
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
<style>
body {
    background:
        linear-gradient(0deg, rgba(0,255,247,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,255,247,0.04) 1px, transparent 1px),
        radial-gradient(circle at center, #00131f 0%, #000 75%);
    background-size: 50px 50px, 50px 50px, cover;
    color: #00fff7;
    font-family: 'Orbitron', sans-serif;
}

.hud {
    background: rgba(0, 18, 30, 0.92);
    border: 1px solid rgba(0,255,247,0.45);
    border-radius: 22px;
    padding: 24px;
    box-shadow: 0 0 60px rgba(0,255,247,0.45);
}

.title {
    text-align:center;
    letter-spacing:6px;
    font-size:34px;
}

.subtitle {
    text-align:center;
    opacity:0.85;
}

.status-bar {
    display:flex;
    justify-content:space-between;
    margin-top:10px;
}

.led {
    width:12px;
    height:12px;
    border-radius:50%;
    background:#00ff6a;
    box-shadow:0 0 12px #00ff6a;
    animation:pulse 1.5s infinite;
}

@keyframes pulse {
    0% {opacity:0.6;}
    50% {opacity:1;}
    100% {opacity:0.6;}
}

.radar {
    position: relative;
    width:160px;
    height:160px;
    border-radius:50%;
    border:1px solid rgba(0,255,247,0.4);
    box-shadow:0 0 30px rgba(0,255,247,0.4);
    background:radial-gradient(circle, rgba(0,255,247,0.15), transparent 70%);
}

.radar::after {
    content:"";
    position:absolute;
    width:2px;
    height:50%;
    background:#00fff7;
    top:0;
    left:50%;
    transform-origin:bottom;
    animation:sweep 2.8s linear infinite;
}

@keyframes sweep {
    from {transform:rotate(0deg);}
    to {transform:rotate(360deg);}
}
</style>

<div class="hud">
    <div class="title">AERO GUARDIAN</div>
    <div class="subtitle">
        Autonomous Dual Drone System for Survivor Detection & Aid Delivery
    <div class="subtitle">
        By 
    <div class="subtitle">
        Nirmiti | Sumit | Onkar
    </div>

    <div class="status-bar">
        <div>System Status: <b>ONLINE</b></div>
        <div class="led"></div>
    </div>
</div>

<div style="
    display:flex;
    flex-direction:column;
    align-items:center;
    justify-content:center;
    margin-top:30px;
">

    <!-- CENTER RADAR -->
    <div class="radar"></div>

    <!-- OPTIONAL DRONE BELOW RADAR -->
    <lottie-player 
        src="https://assets9.lottiefiles.com/packages/lf20_kkflmtur.json"
        style="width:140px;height:140px; margin-top:20px;"
        loop autoplay>
    </lottie-player>

</div>


<script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
""")

    mode = gr.Radio(
        ["Upload Image", "Go Live (Camera)"],
        value="Upload Image",
        label="INPUT MODE"
    )

    upload = gr.Image(
        type="pil",
        sources=["upload"],
        label="DRONE IMAGE INPUT"
    )

    live = gr.Image(
        type="pil",
        sources=["webcam"],
        visible=False,
        label="LIVE CAMERA"
    )

    output = gr.Image(label="TARGET FEED")
    log = gr.Textbox(label="AI COMMAND LOG", lines=6)
    counter = gr.Textbox(label="SURVIVORS DETECTED")

    scan = gr.Button("🚨 INITIATE SCAN")

    def switch(choice):
        return (
            gr.update(visible=choice == "Upload Image"),
            gr.update(visible=choice == "Go Live (Camera)")
        )

    mode.change(switch, mode, [upload, live])
    scan.click(detect, upload, [output, log, counter])
    live.change(detect, live, [output, log, counter])

demo.launch(
    server_name="0.0.0.0",
    server_port=7860
)
