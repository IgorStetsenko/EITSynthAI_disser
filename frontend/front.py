import base64
import os
import glob
import requests
import streamlit as st
import time
import io
import subprocess
import platform

from datetime import datetime
import frontend_config as config

from PIL import Image
from loguru import logger
from frontend_utils import dicom_sequence_to_zip, dicom_sequence_custom_to_zip, dicom_frame_to_zip, \
    image_axial_slice_to_zip, nii_sequence_to_zip, add_log

log_path = "logs/{time:YYYY}/{time:MM}/{time:DD}/"
os.makedirs(os.path.dirname(log_path.format(time=datetime.now())), exist_ok=True)

# ============================================
# ЖЕСТКО ЗАДАНЫЕ ПУТИ (абсолютные для Docker)
# ============================================
RESULTS_DIR = "/app/generation_results"
RECON_FRAMES_DIR = "/app/generation_results/recon_frames_memory_safe"
RECON_GIF_PATH = "/app/generation_results/eit_reconstruction_memory_safe.gif"

# ============================================
# Инициализация session_state для флагов показа
# ============================================
if "show_frames_flag" not in st.session_state:
    st.session_state.show_frames_flag = False
if "show_gif_flag" not in st.session_state:
    st.session_state.show_gif_flag = False
if "recon_result" not in st.session_state:
    st.session_state.recon_result = None  # сюда сохраняем результат реконструкции

# ============================================
# Вспомогательная функция с детальной диагностикой
# ============================================
def _show_frames(frames_dir: str):
    """Отображает серию кадров из указанной директории."""
    
    # Детальная диагностика
    st.info(f"🔍 Проверка пути: `{frames_dir}`")
    st.write(f"- Существует: {os.path.exists(frames_dir)}")
    st.write(f"- Это директория: {os.path.isdir(frames_dir) if os.path.exists(frames_dir) else 'N/A'}")
    
    if os.path.exists(frames_dir):
        try:
            all_files = os.listdir(frames_dir)
            st.write(f"- Все файлы ({len(all_files)}): {all_files[:5]}...")
            png_files = [f for f in all_files if f.lower().endswith('.png')]
            st.write(f"- PNG файлы ({len(png_files)}): {png_files[:5]}...")
        except Exception as e:
            st.error(f"❌ Ошибка чтения директории: {str(e)}")
            return
    else:
        st.error(f"❌ Директория не существует!")
        parent = os.path.dirname(frames_dir)
        st.write(f"📂 Родительская директория `{parent}` существует: {os.path.exists(parent)}")
        if os.path.exists(parent):
            try:
                st.write(f"📄 Содержимое родителя: {os.listdir(parent)}")
            except Exception as e:
                st.write(f"⚠️ Не удалось прочитать: {str(e)}")
        return

    frames_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith('.png')])

    if not frames_files:
        st.warning("ℹ️ В папке нет PNG-файлов.")
        return

    st.success(f"✅ Найдено кадров: {len(frames_files)}")
    st.subheader(f"🖼️ Кадры ({len(frames_files)})")

    # Превью первых 6 кадров
    cols = st.columns(min(6, len(frames_files)))
    for idx, col in enumerate(cols):
        if idx < len(frames_files):
            try:
                img_path = os.path.join(frames_dir, frames_files[idx])
                img = Image.open(img_path)
                col.image(img, caption=f"Кадр {idx + 1}", use_container_width=True)
            except Exception as e:
                col.error(f"❌ Ошибка: {str(e)[:50]}")

    # Слайдер для просмотра всех кадров
    # ВАЖНО: ключ должен быть стабильным, чтобы значение сохранялось между rerun
    if len(frames_files) > 1:
        slider_key = f"slider_{hash(frames_dir)}"
        selected = st.slider(
            "Выберите кадр",
            0, len(frames_files) - 1,
            value=st.session_state.get(slider_key, 0),
            key=slider_key,
        )
        try:
            img_path = os.path.join(frames_dir, frames_files[selected])
            img = Image.open(img_path)
            img_col, _ = st.columns([1, 1])
            with img_col:
                st.image(img, caption=f"Кадр {selected + 1} ({frames_files[selected]})", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Ошибка загрузки кадра: {str(e)}")

# ============================================
# Настройка страницы
# ============================================
st.set_page_config(page_title="", layout="wide")

# --- Кастомный CSS ---
st.markdown("""
<style>
    .main-title {
        text-align: center;
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        padding: 0.5rem 0 1rem 0;
        background: linear-gradient(90deg, #1a1a2e, #16213e);
        border-radius: 12px;
        margin-bottom: 1rem;
    }

    .mode-card {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    .mode-card h4 {
        margin: 0 0 0.3rem 0;
        color: #4fc3f7;
        font-size: 1.1rem;
    }
    .mode-card p {
        margin: 0;
        color: #b0bec5;
        font-size: 0.88rem;
        line-height: 1.45;
    }
    .mode-card.selected {
        border: 2px solid #4fc3f7;
        box-shadow: 0 0 18px rgba(79,195,247,0.25);
    }

    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
        border-radius: 10px;
        padding: 10px 14px;
        margin-bottom: 6px;
        transition: background 0.2s;
    }
    section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
        background: rgba(79,195,247,0.08);
    }

    .results-section {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='main-title'>🫁 Сервис формирования датасета для ЭИТ</div>",
    unsafe_allow_html=True,
)

st.sidebar.image("logo.png", use_container_width=True)

# ============================================
# ВКЛАДКА 1: Генерация датасета
# ============================================
tab1, tab2 = st.tabs(["🔬 Генерация датасета", "🫁 Реконструкция дыхания EIT"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        with st.expander("📖 Описание решения"):
            st.markdown("""
            Сервис позволяет генерировать датасеты для ЭИТ. Перед запуском необходимо выбрать режим генерации
            и загрузить соответствующий файл. Сервис поддерживает файлы **.dicom**, **.nii**.
            """)
    with col2:
        with st.expander("⚙️ Описание режимов генерации датасета для ЭИТ"):
            st.markdown("""
            * **dicom_sequences_auto** — Автоматический режим.
            * **dicom_sequences_custom** — Ручной режим. Пользователь задаёт номер среза относительно центрального.
            * **dicom_frame** — Обработка одного DICOM-среза.
            * **nii** — Формат файла исследования .nii.
            """)

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🎛️ Режим генерации")

    MODES = {
        "dicom_sequences_auto":   {"icon": "🤖", "title": "DICOM Авто",   "desc": "Автоматический выбор среза из серии"},
        "dicom_sequences_custom": {"icon": "🎯", "title": "DICOM Ручной", "desc": "Выбор среза вручную относительно центра"},
        "dicom_frame":            {"icon": "🖼️", "title": "DICOM Кадр",  "desc": "Обработка одного DICOM-среза"},
        "nii":                    {"icon": "📦", "title": "NIfTI",        "desc": "Формат .nii"},
    }

    generation_mode = st.sidebar.radio(
        "Выберите режим генерации датасета:",
        options=list(MODES.keys()),
        format_func=lambda k: f"{MODES[k]['icon']}  {MODES[k]['title']}",
        label_visibility="collapsed",
    )

    sel = MODES[generation_mode]
    st.sidebar.markdown(
        f"""
        <div class="mode-card selected">
            <h4>{sel['icon']} {sel['title']}</h4>
            <p>{sel['desc']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if generation_mode == "dicom_sequences_custom":
        custom_input = st.sidebar.text_input(
            "Номер среза относительно центрального (+1,+2,-1,-2):"
        )

    if __name__ == "__main__":
        uploaded_file = st.file_uploader("📁 Загрузите файл", accept_multiple_files=True)
        button_flag = st.button("🚀 Запустить генерацию датасета для ЭИТ", type="primary")

        def _handle_response(response, t_finish):
            if response.status_code == 200:
                result = response.json()
                st.success(f"✅ Обработка завершена за {result.get('execution_time', int(t_finish))} с")
                st.success(f"⏱ Время сегментации: {result['segmentation_time']} c")
                st.success(f"⏱ Время генерации синтетического датасета: {result['simulation_time']} c")
                st.success(f"💾 Синтетический датасет: {result['saved_file_name']}")

                if 'text_data' in result:
                    st.subheader("Визуализация результатов сегментации:")
                    st.text(result['text_data'])

                if 'image' in result:
                    img_bytes = base64.b64decode(result['image'].encode('utf-8'))
                    img = Image.open(io.BytesIO(img_bytes))
                    st.image(img, caption="Визуализация результатов сегментации", use_container_width=True)
            else:
                st.error(f"❌ Ошибка обработки: {response.text}")

        if button_flag and uploaded_file is not None:
            st.write("📄 Файл успешно загружен!")
            with st.spinner('⏳ Обработка файлов...'):
                add_log(log_path, generation_mode, 'INFO')
                try:
                    if generation_mode == "dicom_sequences_auto":
                        file_zip = dicom_sequence_to_zip(uploaded_file)
                        url = config.upload_dicom_sequence_http
                    elif generation_mode == "dicom_sequences_custom":
                        file_zip = dicom_sequence_custom_to_zip(uploaded_file, custom_input)
                        url = config.upload_dicom_sequence_custom_http
                    elif generation_mode == "dicom_frame":
                        file_zip = dicom_frame_to_zip(uploaded_file)
                        url = config.upload_dicom_frame_http
                    elif generation_mode == "nii":
                        file_zip = nii_sequence_to_zip(uploaded_file)
                        url = config.upload_nii_http
                    else:
                        st.error("❌ Неизвестный режим генерации")
                        file_zip = None
                        url = None

                    if file_zip and url:
                        files = {'file': ('data.zip', file_zip.getvalue(), 'application/zip')}
                        t_start = time.time()
                        response = requests.post(url, files=files)
                        t_finish = time.time() - t_start
                        _handle_response(response, t_finish)

                except requests.exceptions.RequestException as e:
                    st.error(f"🔌 Ошибка соединения с сервером: {str(e)}")
                except Exception as e:
                    st.error(f"💥 Неожиданная ошибка: {str(e)}")

# ============================================
# ВКЛАДКА 2: Реконструкция дыхания EIT
# ============================================
with tab2:
    st.markdown("### 🫁 Реконструкция дыхания EIT")

    run_reconstruct = st.button("🚀 Запустить реконструкцию", type="primary")

    if run_reconstruct:
        with st.spinner('⏳ Выполняется реконструкция...'):
            try:
                response = requests.post(config.reconstruct_http, timeout=600)

                if response.status_code == 200:
                    result = response.json()
                    # Сохраняем результат в session_state, чтобы он пережил rerun
                    st.session_state.recon_result = result
                    st.success("✅ Реконструкция завершена!")
                else:
                    st.error(f"❌ Ошибка: {response.text}")

            except Exception as e:
                st.error(f"💥 Ошибка: {str(e)}")

    # Отображаем сохранённый результат реконструкции (если есть)
    if st.session_state.recon_result is not None:
        result = st.session_state.recon_result
        gif_bytes = base64.b64decode(result['gif'].encode('utf-8'))
        gif_col, _ = st.columns([1, 1])
        with gif_col:
            st.image(gif_bytes, caption="Анимация реконструкции", use_container_width=True)

        frames_dir = result.get('frames_dir', RECON_FRAMES_DIR)
        _show_frames(frames_dir)

    # ============================================
    # Готовые результаты (без повторной обработки)
    # ============================================
    st.markdown("---")
    st.markdown(
        """
        <div class="results-section">
            <h4 style="margin:0 0 0.3rem 0; color:#4fc3f7;">📂 Готовые результаты моделирования</h4>
            <p style="margin:0 0 1rem 0; color:#90a4ae; font-size:0.9rem;">
                Отобразите результаты уже выполненной реконструкции без повторного запуска.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        .buttons-container {
            margin: 1.5rem 0 1.5rem 0;
            padding: 1rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown('<div class="buttons-container">', unsafe_allow_html=True)
    
    col_btn1, col_btn2, col_btn3, _ = st.columns([1, 1, 1, 1])

    with col_btn1:
        # Кнопка теперь лишь переключает флаг в session_state
        if st.button(
            "🖼️ Показать серию кадров",
            help=f"Отобразить все PNG-кадры из папки `{RECON_FRAMES_DIR}`",
        ):
            st.session_state.show_frames_flag = True
    with col_btn2:
        if st.button(
            "🎬 Показать GIF анимацию",
            help=f"Отобразить готовый GIF-файл `{RECON_GIF_PATH}`",
        ):
            st.session_state.show_gif_flag = True
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Рендерим контент по флагам — он будет жить между rerun от слайдера
    if st.session_state.show_frames_flag:
        _show_frames(RECON_FRAMES_DIR)

    if st.session_state.show_gif_flag:
        if os.path.exists(RECON_GIF_PATH):
            try:
                gif_col, _ = st.columns([1, 1])
                with gif_col:
                    st.image(RECON_GIF_PATH, caption="GIF анимация реконструкции", use_container_width=True)
                st.success(f"✅ GIF загружен: {os.path.getsize(RECON_GIF_PATH) / 1024 / 1024:.2f} MB")
            except Exception as e:
                st.error(f"❌ Ошибка загрузки GIF: {str(e)}")
        else:
            st.warning(f"⚠️ GIF-файл не найден: `{RECON_GIF_PATH}`")
            if os.path.exists(RESULTS_DIR):
                gif_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.gif')]
                if gif_files:
                    st.info(f"🎬 Найденные GIF файлы: {gif_files}")