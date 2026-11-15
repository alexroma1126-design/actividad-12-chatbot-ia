# Actividad 12 · Chatbot Streamlit con OpenAI GPT-OSS a través de Groq

**Asignatura:** Fundamentos de Inteligencia Artificial  
**Consigna:** *Building a Simple Chatbot with Streamlit and Hugging Face or OpenAI Models*.

Esta edición utiliza el modelo abierto **`openai/gpt-oss-20b`**, desarrollado por OpenAI y alojado por **GroqCloud**. No utiliza la API de pago de OpenAI. El acceso al plan gratuito de Groq está sujeto a límites de solicitudes y tokens. El código usa el SDK `openai` como cliente compatible hacia una URL de Groq: `https://api.groq.com/openai/v1`.

## Funcionalidades

- Chat general en español con historial.
- Subida de CSV (hasta 3 MB) y vista previa.
- Preguntas fundamentadas en el CSV, traducidas a SQL por GPT-OSS y ejecutadas en SQLite **en memoria**.
- Si la pregunta exige datos ausentes, se pide al modelo rechazarla.
- SQL de solo lectura autorizado, con un máximo de 100 filas mostradas.
- Panel para mostrar la consulta SQL y explicar los resultados.

**Limitaciones:** una IA puede equivocarse al identificar el alcance o escribir SQL; prueba los resultados antes de entregarlos. El esquema y algunas categorías cortas del CSV se comparten con Groq para interpretar preguntas. No utilices CSV con datos sensibles.

## 1. Configurar tu clave (sin publicarla)

1. Obtén una clave en https://console.groq.com/keys; no hace falta pagar para iniciar con el plan gratuito (límites aplicables).
2. En la carpeta raíz crea `.streamlit/secrets.toml` (ya existe `.streamlit/secrets.toml.example` como ejemplo).
3. Escribe **solo en tu equipo**:

   ```toml
   GROQ_API_KEY = "TU_CLAVE_PRIVADA"
   ```

4. Guarda. No publiques, envíes ni muestres esta clave en capturas. El `.gitignore` excluye `secrets.toml`.

## 2. Ejecutar en VS Code / PowerShell (Windows)

Abre la carpeta `Actividad12_Groq_Gratis` en VS Code; en su terminal:

```powershell
& "D:\venvs\a12\Scripts\Activate.ps1"
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m streamlit run app.py
```

Si `D:\venvs\a12` no existe, crea un entorno nuevo en una ruta corta para evitar **WinError 206**:

```powershell
New-Item -ItemType Directory -Force -Path "D:\venvs"
python -m venv "D:\venvs\a12"
& "D:\venvs\a12\Scripts\Activate.ps1"
```

Abre http://localhost:8501. Si el puerto ya está ocupado, detén el Streamlit anterior con `Ctrl+C` o usa `--server.port 8502`.

## 3. Demostración para el docente

Chat general: `Explícame la diferencia entre aprendizaje supervisado y no supervisado.`

En modo **Preguntas sobre un CSV**, carga `datos/clientes_ejemplo.csv` y pregunta:

1. `¿Cuántos clientes hay en total?` → cuenta todas las filas.
2. `¿Cuántos clientes hay por género?` → agrupación por columna.
3. `¿Cuál es la edad promedio?` → promedio calculado.
4. `¿Cuál es la capital de Francia?` → debe responder que está fuera del CSV.
5. `¿Cuál es el mejor canal de publicidad?` → debe rechazar porque el dataset no tiene esa columna.

Abre el botón **Ver consulta SQL utilizada** y compara con los datos cargados.

## 4. Publicar en GitHub y Streamlit Community Cloud

1. Subir `app.py`, `requirements.txt`, `.gitignore`, `README.md`, `datos/`, `tests/` y opcionalmente `.streamlit/secrets.toml.example`.
2. **Jamás subas `.streamlit/secrets.toml`.** Verifica antes: `git status --ignored`.
3. Visita https://share.streamlit.io e indica repositorio, rama y `app.py`.
4. En **Advanced settings → Secrets**, pega privadamente:

   ```toml
   GROQ_API_KEY = "TU_CLAVE_PRIVADA"
   ```

5. Despliega, valida la URL pública y envíala a Moodle junto con `app.py`.

**Aviso:** cualquier persona que use una app pública puede agotar los límites gratuitos de tu clave del servidor. No actives facturación ni plan de pago si solo deseas utilizar el plan Free.

## Referencias

- Notebook de clase: `OpenAI_API.ipynb`, actividad final.
- Compatibilidad API con cliente OpenAI: https://console.groq.com/docs/openai
- Modelo: https://console.groq.com/docs/model/openai/gpt-oss-20b
- Salidas estructuradas JSON Schema: https://console.groq.com/docs/structured-outputs
- Publicación: https://docs.streamlit.io/deploy/streamlit-community-cloud
