import os
from functools import lru_cache

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model


MODEL_PATH = os.getenv("INSWAPPER_MODEL", os.path.join("models", "inswapper_128.onnx"))
PROVIDERS = ["CPUExecutionProvider"]

app = FastAPI(title="Xedoc Hands FaceSwap Bridge")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://hands.xedoc.ru",
        "http://hands.xedoc.ru",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_face_app() -> FaceAnalysis:
    face_app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    return face_app


@lru_cache(maxsize=1)
def get_swapper():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"FaceSwap model was not found: {MODEL_PATH}. "
            "Put inswapper_128.onnx into models/ or set INSWAPPER_MODEL."
        )

    return get_model(MODEL_PATH, providers=PROVIDERS)


def decode_image(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    return image


def biggest_face(faces):
    if not faces:
        return None

    return max(faces, key=lambda face: (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1]))


@app.get("/health")
def health():
    model_ready = os.path.exists(MODEL_PATH)
    return {"ok": True, "modelReady": model_ready, "modelPath": MODEL_PATH}


@app.post("/swap")
async def swap(frame: UploadFile = File(...), source: UploadFile = File(...)):
    try:
        face_app = get_face_app()
        swapper = get_swapper()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    frame_image = decode_image(await frame.read())
    source_image = decode_image(await source.read())

    source_face = biggest_face(face_app.get(source_image))
    target_face = biggest_face(face_app.get(frame_image))

    if source_face is None:
        raise HTTPException(status_code=422, detail="No source face found")

    if target_face is None:
        raise HTTPException(status_code=422, detail="No target face found")

    swapped = swapper.get(frame_image, target_face, source_face, paste_back=True)
    success, encoded = cv2.imencode(".jpg", swapped, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

    if not success:
        raise HTTPException(status_code=500, detail="Could not encode swapped frame")

    return Response(content=encoded.tobytes(), media_type="image/jpeg")
