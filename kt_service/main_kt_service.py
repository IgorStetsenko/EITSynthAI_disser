import logging
import numpy
import zipfile
import sys
import subprocess
import base64
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from io import BytesIO
from PIL import Image
from .ai_tools.ai_tools import DICOMSequencesToMask, DICOMSequencesToMaskCustom, DICOMToMask, NIIToMask
from pathlib import Path

# Добавляем папку `kt-service` в PYTHONPATH
sys.path.append(str(Path(__file__).parent))

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

dicom_seq_to_mask = DICOMSequencesToMask()
dicom_seq_to_mask_custom = DICOMSequencesToMaskCustom()
dicom_seq_to_mask_frame = DICOMToMask()
nii_seq_to_mask = NIIToMask()

logger.info("🚀 Запущен main_kt_service 🚀")

# Пути
SCRIPTS_DIR = "/app/kt_service/scripts"
RESULTS_DIR = "/app/generation_results"

@app.post("/uploadDicomSequence")
async def upload_file(file: UploadFile = File(...)):
    try:
        logger.info("✅ Запущен метод uploadDicomSequence")
        contents = await file.read()
        zip_buffer = BytesIO(contents)
        answer = dicom_seq_to_mask.get_coordinate_slice_from_dicom(zip_buffer)
        return answer
    except zipfile.BadZipFile:
        logger.error("🔴 Загруженный файл не является корректным ZIP-архивом")
        raise HTTPException(status_code=400, detail="Загруженный файл не является корректным ZIP-архивом")
    except Exception as e:
        logger.error(f"🔴 Ошибка при обработке файла: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла: {str(e)}")

@app.post("/uploadDicomSequenceCustom")
async def upload_file(file: UploadFile = File(...)):
    try:
        logger.info("✅ Запущен метод uploadDicomSequenceCustom")
        contents = await file.read()
        zip_buffer = BytesIO(contents)
        custom_number_slise = 0
        answer = dicom_seq_to_mask_custom.get_coordinate_slice_from_dicom_custom(zip_buffer)
        return answer
    except zipfile.BadZipFile:
        logger.error(" Загруженный файл не является корректным ZIP-архивом")
        raise HTTPException(status_code=400, detail="Загруженный файл не является корректным ZIP-архивом")
    except Exception as e:
        logger.error(f"🔴 Ошибка при обработке файла: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла: {str(e)}")

@app.post("/uploadDicomFrame")
async def upload_file(file: UploadFile = File(...)):
    try:
        logger.info("✅ Запущен метод uploadDicomFrame")
        contents = await file.read()
        zip_buffer = BytesIO(contents)
        custom_number_slise = 0
        answer = dicom_seq_to_mask_frame.get_coordinate_slice_from_dicom_frame(zip_buffer)
        return answer
    except zipfile.BadZipFile:
        logger.error("🔴 Загруженный файл не является корректным ZIP-архивом")
        raise HTTPException(status_code=400, detail="Загруженный файл не является корректным ZIP-архивом")
    except Exception as e:
        logger.error(f"🔴 Ошибка при обработке файла: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла: {str(e)}")

@app.post("/uploadNII")
async def upload_file(file: UploadFile = File(...)):
    try:
        logger.info("✅ Запущен метод uploadNII")
        contents = await file.read()
        zip_buffer = BytesIO(contents)
        answer = nii_seq_to_mask.get_coordinate_slice_from_nii(zip_buffer)
        return answer
    except zipfile.BadZipFile:
        logger.error("🔴 Загруженный файл не является корректным ZIP-архивом")
        raise HTTPException(status_code=400, detail="Загруженный файл не является корректным ZIP-архивом")
    except Exception as e:
        logger.error(f"🔴 Ошибка при обработке файла: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла: {str(e)}")

@app.post("/reconstruct")
async def run_reconstruction():
    """Запуск реконструкции дыхания EIT"""
    try:
        logger.info("✅ Запущен метод reconstruct")
        
        results_dir = "/app/generation_results"
        script_path = "/app/kt_service/scripts/reconstruct_breath.py"
        
        # Запускаем скрипт
        result = subprocess.run(
            ["python3", script_path, results_dir],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Ошибка: {result.stderr}")
        
        # Читаем GIF
        gif_path = os.path.join(results_dir, "eit_reconstruction_memory_safe.gif")
        frames_dir = os.path.join(results_dir, "recon_frames_memory_safe")
        
        if not os.path.exists(gif_path):
            raise HTTPException(status_code=500, detail="GIF не создан")
        
        with open(gif_path, "rb") as f:
            gif_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Получаем список кадров
        frames_list = []
        if os.path.exists(frames_dir):
            frames_list = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        
        return {
            "status": "success",
            "gif": gif_data,
            "frames_count": len(frames_list),
            "frames_dir": frames_dir
        }
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Превышено время (10 мин)")
    except Exception as e:
        logger.error(f"🔴 Ошибка: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))