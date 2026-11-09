from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = FastAPI()

# --- МОДУЛЬ РАБОТЫ С МОДЕЛЬЮ (встроен) ---
try:
    model = load_model('iris_model.keras')
    scaler = joblib.load('scaler.pkl')
    class_names = ['setosa (щетинистый)', 'versicolor (разноцветный)', 'virginica (виргинский)']
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    model = None
    scaler = None


def predict_iris(sepal_length, sepal_width, petal_length, petal_width):
    if model is None or scaler is None:
        return "Модель не загружена"
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    class_idx = np.argmax(prediction)
    return class_names[class_idx]


# --- ВЕБ-ЧАСТЬ ---
def get_html_page(sl="5.1", sw="3.5", pl="1.4", pw="0.2", result_block=""):
    # Используем f-строку. Все фигурные скобки в HTML теперь безопасны.
    return f"""
    <!DOCTYPE html>
    <html lang="ru">
    <head>
        <meta charset="UTF-8">
        <title>Классификатор Ирисов</title>
        <style>
            body {{ font-family: Arial, sans-serif; max-width: 500px; margin: 50px auto; padding: 20px; border: 1px solid #ccc; border-radius: 10px; background-color: #f9f9f9; }}
            .form-group {{ margin-bottom: 15px; }}
            label {{ display: block; margin-bottom: 5px; font-weight: bold; }}
            input[type="number"] {{ width: 100%; padding: 8px; box-sizing: border-box; }}
            button {{ padding: 10px 15px; background-color: #28a745; color: white; border: none; border-radius: 5px; cursor: pointer; width: 100%; }}
            .result {{ margin-top: 20px; padding: 15px; background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; color: #155724; text-align: center; }}
        </style>
    </head>
    <body>
        <h2>Введите параметры ириса</h2>
        <form method="post" action="/">
            <div class="form-group">
                <label>Длина чашелистика (см):</label>
                <input type="number" step="0.1" name="sepal_length" value="{sl}" required>
            </div>
            <div class="form-group">
                <label>Ширина чашелистика (см):</label>
                <input type="number" step="0.1" name="sepal_width" value="{sw}" required>
            </div>
            <div class="form-group">
                <label>Длина лепестка (см):</label>
                <input type="number" step="0.1" name="petal_length" value="{pl}" required>
            </div>
            <div class="form-group">
                <label>Ширина лепестка (см):</label>
                <input type="number" step="0.1" name="petal_width" value="{pw}" required>
            </div>
            <button type="submit">Отправить</button>
        </form>
        {result_block}
    </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse)
async def get_form():
    return get_html_page()


@app.post("/", response_class=HTMLResponse)
async def post_form(
        sepal_length: float = Form(...),
        sepal_width: float = Form(...),
        petal_length: float = Form(...),
        petal_width: float = Form(...)
):
    result = predict_iris(sepal_length, sepal_width, petal_length, petal_width)
    result_block = f"<div class='result'><h3>Результат: {result}</h3></div>"

    return get_html_page(
        sl=sepal_length, sw=sepal_width,
        pl=petal_length, pw=petal_width,
        result_block=result_block
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)