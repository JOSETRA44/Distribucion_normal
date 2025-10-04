        "      ██╗ ██████╗ ███████╗███████╗████████╗██████╗  █████╗ ",
        "      ██║██╔═══██╗██╔════╝██╔════╝╚══██╔══╝██╔══██╗██╔══██╗",
        "      ██║██║   ██║███████╗█████╗     ██║   ██████╔╝███████║",
        " ██   ██║██║   ██║╚════██║██╔══╝     ██║   ██╔══██╗██╔══██║",
        " ╚█████╔╝╚██████╔╝███████║███████╗   ██║   ██║  ██║██║  ██║",
        "  ╚════╝  ╚═════╝ ╚══════╝╚══════╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝"
# Calculadora de Distribución Normal (Manual + IA)

Herramienta en Python para calcular probabilidades de una Normal(μ, σ), generar gráficos PDF/CDF y usar IA para interpretar problemas en lenguaje natural, con un chat de seguimiento que aclara dudas y puede generar nuevas gráficas automáticamente.

## Características

- **[Modos]**
  - Manual: introduces μ, σ y el cálculo (entre/menor/mayor).
  - IA (Gemini): pegas tu problema en lenguaje natural; la IA devuelve un JSON con parámetros y el script calcula y grafica.
  - IA (ChatGPT): igual que Gemini pero con OpenAI.
- **[Gráficos]** PDF y CDF con sombreado, bandas 68-95-99.7%, percentiles p50/p90/p95.
- **[Guardado]** Imágenes en la carpeta `imagenes/` (configurable con `IMAGES_DIR` en `.env`).
- **[Chat de seguimiento]** Tras cada ejecución, se inicia un chat que responde dudas con el contexto actual y genera automáticamente una gráfica por cada respuesta (salvo que indiques “sin grafico”).
- **[Comando extra]** En el chat, escribe `grafica` para forzar re-graficar la última duda resuelta.
- **[Sin datos crudos]** El chat evita devolver arreglos; si graficar aplica, la IA responde con un JSON de acción `plot` que el script ejecuta.

## Requisitos

Archivo `requirements.txt` incluido:

```
numpy==2.1.2
scipy==1.14.1
matplotlib==3.9.2
python-dotenv==1.0.1
openai==1.52.2
google-generativeai==0.8.2
```

Instalación (recomendado usar venv):

```powershell
# Windows PowerShell
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

## Configuración (.env)

Crea `c:\Users\USER\source\repos\new\.env` con tus llaves y preferencias. Ejemplo:

```
IMAGES_DIR=imagenes

# IA 2 (Gemini)
GEMINI_API_KEY=tu_gemini_key
GEMINI_MODEL=gemini-1.5-pro-latest

# IA 3 (ChatGPT) - modo principal
OPENAI_API_KEY=tu_openai_key
OPENAI_MODEL=gpt-4o-mini

# Chat de seguimiento persistente (3ra llave)
OPENAI_FOLLOWUP_API_KEY=tu_openai_followup_key
OPENAI_FOLLOWUP_MODEL=gpt-4o-mini
```

Notas:
- Si falta una clave, el script te la pedirá una sola vez y la guardará en `.env`.
- `GEMINI_MODEL`, `OPENAI_MODEL` y `OPENAI_FOLLOWUP_MODEL` son opcionales. El script tiene fallbacks.

## Uso

```powershell
# Ejecutar
python exel.py
```

Sigue el menú:
- 1) Manual
- 2) IA (Gemini)
- 3) IA (ChatGPT)

En los modos IA, pega tu problema (con μ, σ, x, o datos crudos si aplica). El sistema generará resultados y gráficos.

### Chat de seguimiento
- Tras cualquier modo, escribe tu duda en texto. Se mostrará la respuesta y se generará automáticamente una gráfica correspondiente.
- Para omitir gráfica en una duda específica, escribe: `sin grafico` (o `sin gráfico` / `no grafico`).
- Para forzar la gráfica de la última duda: escribe `grafica`.
- Envía Enter vacío para terminar el chat.

## Estructura del proyecto

- `exel.py`: script principal.
- `requirements.txt`: dependencias.
- `README.md`: esta guía.
- `imagenes/`: carpeta de salida de imágenes (se crea automáticamente).
- `.env`: variables de entorno con llaves/modelos.

## Solución de problemas

- [Gemini 404 model not found]
  - Asegura que `GEMINI_MODEL` existe y está habilitado en tu cuenta. Sugerencias:
    - `gemini-1.5-pro-latest`
    - `gemini-1.5-pro`
    - `gemini-1.5-flash-latest`
  - Verifica modelos: https://ai.google.dev/gemini-api/docs/models

- [No arranca el chat de seguimiento]
  - Revisa `OPENAI_FOLLOWUP_API_KEY` en `.env` o deja que el script la solicite.

- [No se guardan imágenes]
  - Verifica permisos de la carpeta `imagenes/` o cambia `IMAGES_DIR`.

## Seguridad
- No compartas `.env` en repos públicos.
- Mantén actualizadas las dependencias.

## Licencia
Este proyecto es de uso libre para fines educativos y de análisis. Ajusta la licencia según tus necesidades.

EL MANTRA... 
