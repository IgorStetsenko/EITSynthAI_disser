import logging
import numpy
import zipfile
import sys


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


@app.post("/uploadDicomSequence")
async def upload_file(file: UploadFile = File(...)):
    try:
        logger.info("✅ Запущен метод uploadDicomSequence")
        contents = await file.read()
        zip_buffer = BytesIO(contents)
        answer = dicom_seq_to_mask.get_coordinate_slice_from_dicom(zip_buffer)
        # Возвращаем JSON с изображением и временем выполнения
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
        # Возвращаем JSON с изображением и временем выполнения
        return answer

    except zipfile.BadZipFile:
        logger.error("Загруженный файл не является корректным ZIP-архивом")
        raise HTTPException(status_code=400, detail="Загруженный файл не является корректным ZIP-архивом")
    except Exception as e:
        logger.error(f"Ошибка при обработке файла: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла: {str(e)}")


@app.post("/uploadDicomFrame")
async def upload_file(file: UploadFile = File(...)):
    try:
        logger.info("✅ Запущен метод uploadDicomFrame")
        contents = await file.read()
        zip_buffer = BytesIO(contents)
        custom_number_slise = 0
        answer = dicom_seq_to_mask_frame.get_coordinate_slice_from_dicom_frame(zip_buffer)
        # Возвращаем JSON с изображением и временем выполнения
        return answer

    except zipfile.BadZipFile:
        logger.error("Загруженный файл не является корректным ZIP-архивом")
        raise HTTPException(status_code=400, detail="Загруженный файл не является корректным ZIP-архивом")
    except Exception as e:
        logger.error(f"Ошибка при обработке файла: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла: {str(e)}")


@app.post("/uploadNII")
async def upload_file(file: UploadFile = File(...)):
    try:
        logger.info("✅ Запущен метод uploadNII")
        contents = await file.read()
        zip_buffer = BytesIO(contents)
        answer = nii_seq_to_mask.get_coordinate_slice_from_nii(zip_buffer)
        # Возвращаем JSON с изображением и временем выполнения
        return answer

    except zipfile.BadZipFile:
        logger.error("Загруженный файл не является корректным ZIP-архивом")
        raise HTTPException(status_code=400, detail="Загруженный файл не является корректным ZIP-архивом")
    except Exception as e:
        logger.error(f"Ошибка при обработке файла: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла: {str(e)}")



@app.post("/reconstruct")
async def upload_file(file: UploadFile = File(...)):
    try:
        logger.info("✅ Запущен метод uploadNII")
        contents = await file.read()
        zip_buffer = BytesIO(contents)
        answer = nii_seq_to_mask.get_coordinate_slice_from_nii(zip_buffer)
        # Возвращаем JSON с изображением и временем выполнения
        return answer

    except zipfile.BadZipFile:
        logger.error("Загруженный файл не является корректным ZIP-архивом")
        raise HTTPException(status_code=400, detail="Загруженный файл не является корректным ZIP-архивом")
    except Exception as e:
        logger.error(f"Ошибка при обработке файла: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла: {str(e)}")