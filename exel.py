import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.ticker import FuncFormatter, MaxNLocator
from typing import Optional
import json

# Cargar variables desde .env si la librería está disponible
try:
    from dotenv import load_dotenv
    load_dotenv()  # no falla si no existe .env
except Exception:
    pass

# =============================
# Utilidades de visualización
# =============================
def adaptive_x_range(mu: float, sigma: float, q_low: float = 0.001, q_high: float = 0.999):
    """
    Devuelve un rango de x adaptativo basado en cuantiles para cubrir casi toda la masa.
    """
    left = norm.ppf(q_low, mu, sigma)
    right = norm.ppf(q_high, mu, sigma)
    # Asegurar un mínimo de ±4σ por robustez ante extremos
    left = min(left, mu - 4 * sigma)
    right = max(right, mu + 4 * sigma)
    # Evitar rangos degenerados si sigma es muy pequeño
    if not np.isfinite(left) or not np.isfinite(right) or left == right:
        left, right = mu - 4 * sigma, mu + 4 * sigma
    return np.linspace(left, right, 1200)

def sigma_bands(ax, mu: float, sigma: float, y_max: float):
    """
    Dibuja bandas de confianza 68-95-99.7% (regla empírica) alrededor de la media.
    """
    bands = [
        (1, (0.10, 0.9), (0.85, 0.20, 0.20), '68% (~1σ)'),
        (2, (0.07, 0.7), (0.20, 0.85, 0.20), '95% (~2σ)'),
        (3, (0.05, 0.5), (0.20, 0.20, 0.85), '99.7% (~3σ)')
    ]
    for k, alpha, color, label in bands:
        ax.axvspan(mu - k * sigma, mu + k * sigma, color=color, alpha=alpha[0], lw=0, label=label)
    # Línea de media
    ax.axvline(mu, color='red', linestyle='--', linewidth=1.5, label=f'Media (μ={mu:g})')

def shade_region_pdf(ax, mu: float, sigma: float, mode: str, x1: Optional[float], x2: Optional[float]):
    """
    Sombrea el área bajo la PDF según el modo: 'entre', 'menor', 'mayor'.
    """
    x = adaptive_x_range(mu, sigma)
    y = norm.pdf(x, mu, sigma)
    if mode == 'entre' and x1 is not None and x2 is not None:
        x_fill = x[(x >= x1) & (x <= x2)]
    elif mode == 'menor' and x1 is not None:
        x_fill = x[x <= x1]
    elif mode == 'mayor' and x1 is not None:
        x_fill = x[x >= x1]
    else:
        x_fill = np.array([])
    if x_fill.size:
        ax.fill_between(x_fill, norm.pdf(x_fill, mu, sigma), color='tab:blue', alpha=0.5, label='Área de interés')
    return x, y

def shade_region_cdf(ax, mu: float, sigma: float, mode: str, x1: Optional[float], x2: Optional[float]):
    """
    Sombrea el área en la CDF con bandas para representar la probabilidad.
    """
    x = adaptive_x_range(mu, sigma)
    c = norm.cdf(x, mu, sigma)
    if mode == 'entre' and x1 is not None and x2 is not None:
        xa = x[(x >= x1) & (x <= x2)]
    elif mode == 'menor' and x1 is not None:
        xa = x[x <= x1]
    elif mode == 'mayor' and x1 is not None:
        xa = x[x >= x1]
    else:
        xa = np.array([])
    if xa.size:
        ax.fill_between(xa, 0, norm.cdf(xa, mu, sigma), step='pre', color='tab:blue', alpha=0.35, label='Acumulado')
    return x, c

def get_gemini_api_key() -> Optional[str]:
    """
    Obtiene la API Key de Gemini desde variables de entorno (.env soportado si existe)
    en este orden de preferencia: GEMINI_API_KEY, GOOGLE_API_KEY, GOOGLE_API_KEY_GEMINI.
    Si no existe, pide al usuario y ofrece guardarla en .env.
    """
    env_keys = [
        os.getenv("GEMINI_API_KEY"),
        os.getenv("GOOGLE_API_KEY"),
        os.getenv("GOOGLE_API_KEY_GEMINI"),
    ]
    for key in env_keys:
        if key and key.strip():
            return key.strip()

    # Pedir al usuario
    api_key = input("Pega tu API Key de Gemini (se puede guardar en .env): ").strip()
    if not api_key:
        return None

    # Ofrecer persistir en .env
    try:
        save = input("¿Guardar esta clave en .env para no pedirla de nuevo? (s/n): ").strip().lower()
        if save == 's':
            env_path = os.path.join(os.getcwd(), '.env')
            # Añade/actualiza de forma simple (append). Si hay duplicado, prevalece la última línea.
            with open(env_path, 'a', encoding='utf-8') as f:
                f.write(f"\nGEMINI_API_KEY={api_key}\n")
            print(f"Clave guardada en {env_path} como GEMINI_API_KEY.")
    except Exception as e:
        print(f"No se pudo guardar la clave en .env: {e}")

    return api_key

def get_images_dir() -> str:
    """
    Devuelve la ruta de la carpeta donde guardar imágenes. Por defecto 'imagenes' en el cwd,
    configurable con la variable de entorno IMAGES_DIR. Crea la carpeta si no existe.
    """
    dir_name = os.getenv("IMAGES_DIR", "imagenes")
    img_dir = os.path.join(os.getcwd(), dir_name)
    try:
        os.makedirs(img_dir, exist_ok=True)
    except Exception as e:
        print(f"No se pudo crear la carpeta de imágenes '{img_dir}': {e}")
    return img_dir

def build_context(mu, sigma, mode, x1, x2, desc, prob, z1, z2, p50, p90, p95, out_path):
    """
    Construye el contexto enviado a ChatGPT para aclarar dudas posteriores.
    """
    return {
        "mu": float(mu),
        "sigma": float(sigma),
        "mode": mode,
        "x1": None if x1 is None else float(x1),
        "x2": None if x2 is None else float(x2),
        "descripcion": desc,
        "probabilidad": float(prob),
        "z1": None if z1 is None else float(z1),
        "z2": None if z2 is None else float(z2),
        "p50": float(p50),
        "p90": float(p90),
        "p95": float(p95),
        "imagen": out_path,
    }

def prompt_followup(context: dict):
    """
    Obsoleto: Se reemplaza por answer_followup_chat_loop().
    """
    # Mantenido por compatibilidad si fuera llamado en otro punto
    answer_followup_chat_loop(context)

def get_followup_openai_api_key() -> Optional[str]:
    """
    API Key dedicada para el chat de seguimiento (tercera llave): OPENAI_FOLLOWUP_API_KEY.
    Se pide una sola vez y se guarda en .env.
    """
    env_keys = [
        os.getenv("OPENAI_FOLLOWUP_API_KEY"),
    ]
    for key in env_keys:
        if key and key.strip():
            return key.strip()

    api_key = input("Pega tu API Key de OpenAI para dudas (se guardará como OPENAI_FOLLOWUP_API_KEY): ").strip()
    if not api_key:
        return None

    try:
        save = 's'  # siempre guardamos esta llave para no volver a pedirla
        if save == 's':
            env_path = os.path.join(os.getcwd(), '.env')
            with open(env_path, 'a', encoding='utf-8') as f:
                f.write(f"\nOPENAI_FOLLOWUP_API_KEY={api_key}\n")
            print(f"Clave de seguimiento guardada en {env_path} como OPENAI_FOLLOWUP_API_KEY.")
    except Exception as e:
        print(f"No se pudo guardar la clave de seguimiento en .env: {e}")

    return api_key

def answer_followup_chat_loop(context: dict) -> None:
    """
    Inicia un chat de seguimiento persistente. Vacío para terminar.
    Puede responder texto o instruir generar nuevas gráficas devolviendo un JSON.
    Contrato JSON opcional:
    {
      "action": "plot",
      "mu": number | null,
      "sigma": number | null,
      "mode": "entre"|"menor"|"mayor" | null,
      "x1": number | null,
      "x2": number | null,
      "save_png": boolean | null,
      "filename": string | null
    }
    Si un campo es null/ausente, se usan valores del contexto.
    """
    api_key = get_followup_openai_api_key()
    if not api_key:
        # Si no hay llave, degradar a la principal de OPENAI como respaldo
        api_key = get_openai_api_key()
        if not api_key:
            return

    try:
        from openai import OpenAI
    except ImportError:
        print("Falta la librería 'openai'. Instálala con: pip install openai")
        return

    preferred_model = os.getenv("OPENAI_FOLLOWUP_MODEL") or os.getenv("OPENAI_MODEL") or os.getenv("AZURE_OPENAI_DEPLOYMENT")
    model_candidates = []
    if preferred_model:
        model_candidates.append(preferred_model)
    model_candidates += [
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-3.5-turbo",
    ]

    client = OpenAI(api_key=api_key)

    system_instructions = (
        "Eres un asistente de estadística que responde en español de forma breve y clara. "
        "No devuelvas arreglos/datos crudos. Si procede graficar, devuelve SOLO un JSON válido siguiendo el contrato indicado. "
        "Si faltan parámetros en el JSON, usa los del contexto proporcionado. Evita inventar datos ajenos al contexto."
    )

    def derive_plot_from_answer(ctx: dict, last_q: str, last_a: str) -> Optional[dict]:
        """Solicita al modelo un JSON de plot (action=plot) basado en la última Q/A y el contexto."""
        last_err_local = None
        for m in model_candidates:
            try:
                resp = client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": system_instructions},
                        {"role": "user", "content": f"Contexto: {json.dumps(ctx, ensure_ascii=False)}"},
                        {"role": "user", "content": f"Última pregunta del usuario: {last_q or ''}"},
                        {"role": "user", "content": f"Última explicación en texto: {last_a or ''}"},
                        {"role": "user", "content": "Devuelve SOLO un JSON (action=plot) que represente exactamente esa duda resuelta. Si faltan parámetros, usa el contexto."},
                    ],
                    temperature=0,
                )
                content_plot = (resp.choices[0].message.content or '').strip()
                text_plot = content_plot
                if text_plot.startswith("```"):
                    text_plot = text_plot.strip('`').strip()
                if text_plot.lower().startswith("json"):
                    text_plot = text_plot[4:].strip()
                payload = json.loads(text_plot)
                if isinstance(payload, dict) and payload.get("action") == "plot":
                    return payload
            except Exception as e:
                last_err_local = e
                continue
        return None

    # Bucle de dudas
    last_text_answer = None
    last_user_question = None
    while True:
        question = input("\nDuda (Enter para terminar, escribe 'grafica' para graficar la duda anterior): ").strip()
        if not question:
            break

        # Comando especial para graficar la última duda resuelta
        if question.lower() == 'grafica':
            payload = None
            # Siempre derivar un JSON de plot fresco basado en la última pregunta/respuesta
            if not last_text_answer:
                print("No hay una duda previa para graficar. Formula una duda primero.")
                continue
            last_err = None
            content_plot = None
            used_model = None
            for m in model_candidates:
                try:
                    resp = client.chat.completions.create(
                        model=m,
                        messages=[
                            {"role": "system", "content": system_instructions},
                            {"role": "user", "content": f"Contexto: {json.dumps(context, ensure_ascii=False)}"},
                            {"role": "user", "content": f"Última pregunta del usuario: {last_user_question or ''}"},
                            {"role": "user", "content": f"Última explicación en texto: {last_text_answer or ''}"},
                            {"role": "user", "content": "Con base en la última explicación, devuelve SOLO un JSON (action=plot) que represente exactamente esa duda resuelta. Si faltan parámetros, usa el contexto."},
                        ],
                        temperature=0,
                    )
                    content_plot = (resp.choices[0].message.content or '').strip()
                    used_model = m
                    break
                except Exception as e:
                    last_err = e
                    continue
            if content_plot is None:
                print("No se pudo obtener instrucciones de graficado de la IA.")
                if last_err:
                    print(f"Último error: {last_err}")
                continue
            text_plot = content_plot
            if text_plot.startswith("```"):
                text_plot = text_plot.strip('`').strip()
            if text_plot.lower().startswith("json"):
                text_plot = text_plot[4:].strip()
            try:
                payload = json.loads(text_plot)
            except Exception:
                payload = None

            if isinstance(payload, dict) and payload.get("action") == "plot":
                mu = payload.get("mu", context.get("mu"))
                sigma = payload.get("sigma", context.get("sigma"))
                mode = payload.get("mode", context.get("mode"))
                x1 = payload.get("x1", context.get("x1"))
                x2 = payload.get("x2", context.get("x2"))
                save_png = bool(payload.get("save_png") or False)
                filename = payload.get("filename") or "followup_plot"
                if mu is not None and sigma is not None and mode in {"entre","menor","mayor"}:
                    plot_and_report(float(mu), float(sigma), mode,
                                    None if x1 is None else float(x1), None if x2 is None else float(x2),
                                    ask_save=(not save_png), suggested_filename=(filename if save_png else None))
                    context.update({
                        "mu": float(mu), "sigma": float(sigma), "mode": mode,
                        "x1": None if x1 is None else float(x1),
                        "x2": None if x2 is None else float(x2),
                    })
                    continue
                else:
                    print("La instrucción de graficado es incompleta. Proporciona más detalles en una duda.")
                    continue

        # Flujo normal de duda (texto o JSON de plot)
        last_err = None
        content = None
        used_model = None
        for m in model_candidates:
            try:
                resp = client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": system_instructions},
                        {"role": "user", "content": f"Contexto: {json.dumps(context, ensure_ascii=False)}"},
                        {"role": "user", "content": f"Pregunta: {question}"},
                        {"role": "user", "content": "Si deseas que genere una nueva gráfica, devuelve solo un JSON válido siguiendo el contrato. Si no, responde en texto."},
                    ],
                    temperature=0,
                )
                content = (resp.choices[0].message.content or '').strip()
                used_model = m
                break
            except Exception as e:
                last_err = e
                continue

        if content is None:
            print("No se pudo obtener respuesta del asistente de seguimiento.")
            if last_err:
                print(f"Último error: {last_err}")
            continue

        text = content
        if text.startswith("```"):
            text = text.strip('`').strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

        did_plot = False
        try:
            payload = json.loads(text)
            if isinstance(payload, dict) and payload.get("action") == "plot":
                mu = payload.get("mu", context.get("mu"))
                sigma = payload.get("sigma", context.get("sigma"))
                mode = payload.get("mode", context.get("mode"))
                x1 = payload.get("x1", context.get("x1"))
                x2 = payload.get("x2", context.get("x2"))
                save_png = bool(payload.get("save_png") or False)
                filename = payload.get("filename") or "followup_plot"
                if mu is not None and sigma is not None and mode in {"entre","menor","mayor"}:
                    plot_and_report(float(mu), float(sigma), mode,
                                    None if x1 is None else float(x1), None if x2 is None else float(x2),
                                    ask_save=(not save_png), suggested_filename=(filename if save_png else None))
                    context.update({
                        "mu": float(mu), "sigma": float(sigma), "mode": mode,
                        "x1": None if x1 is None else float(x1),
                        "x2": None if x2 is None else float(x2),
                    })
                    last_plot_payload = payload
                    did_plot = True
        except Exception:
            pass

        if not did_plot:
            print(f"\nRespuesta ({used_model or 'OpenAI'}):\n{content}\n")
            last_text_answer = content
            last_user_question = question
            # Al haber nueva respuesta textual, invalidar cualquier instrucción de plot previa
            # Auto-graficar salvo que el usuario pida explícitamente no graficar
            if question.lower().strip() not in {"sin grafico", "sin gráfico", "no grafico", "no gráfico"}:
                payload_auto = derive_plot_from_answer(context, last_user_question, last_text_answer)
                if isinstance(payload_auto, dict) and payload_auto.get("action") == "plot":
                    mu = payload_auto.get("mu", context.get("mu"))
                    sigma = payload_auto.get("sigma", context.get("sigma"))
                    mode = payload_auto.get("mode", context.get("mode"))
                    x1 = payload_auto.get("x1", context.get("x1"))
                    x2 = payload_auto.get("x2", context.get("x2"))
                    save_png = bool(payload_auto.get("save_png") or False)
                    filename = payload_auto.get("filename") or "followup_plot"
                    if mu is not None and sigma is not None and mode in {"entre","menor","mayor"}:
                        plot_and_report(float(mu), float(sigma), mode,
                                        None if x1 is None else float(x1), None if x2 is None else float(x2),
                                        ask_save=(not save_png), suggested_filename=(filename if save_png else None),
                                        start_followup=False)

def get_openai_api_key() -> Optional[str]:
    """
    Obtiene la API Key de OpenAI (ChatGPT) desde variables de entorno (.env soportado si existe)
    en este orden de preferencia: OPENAI_API_KEY, AZURE_OPENAI_API_KEY.
    Si no existe, la pide y ofrece guardarla en .env.
    """
    env_keys = [
        os.getenv("OPENAI_API_KEY"),
        os.getenv("AZURE_OPENAI_API_KEY"),
    ]
    for key in env_keys:
        if key and key.strip():
            return key.strip()

    api_key = input("Pega tu API Key de OpenAI (se puede guardar en .env): ").strip()
    if not api_key:
        return None

    try:
        save = input("¿Guardar esta clave en .env para no pedirla de nuevo? (s/n): ").strip().lower()
        if save == 's':
            env_path = os.path.join(os.getcwd(), '.env')
            with open(env_path, 'a', encoding='utf-8') as f:
                f.write(f"\nOPENAI_API_KEY={api_key}\n")
            print(f"Clave guardada en {env_path} como OPENAI_API_KEY.")
    except Exception as e:
        print(f"No se pudo guardar la clave en .env: {e}")

    return api_key

def plot_and_report(mu: float, sigma: float, mode: str, x1: Optional[float] = None, x2: Optional[float] = None,
                    ask_save: bool = True, suggested_filename: Optional[str] = None,
                    start_followup: bool = True):
    """
    Calcula, imprime interpretaciones y genera gráficos PDF/CDF.
    """
    # Validación
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
        print("\nError: μ debe ser finito y σ debe ser un número positivo.")
        return

    # Cálculos de probabilidad
    if mode == 'entre' and x1 is not None and x2 is not None:
        p1 = norm.cdf(x1, mu, sigma)
        p2 = norm.cdf(x2, mu, sigma)
        prob = max(0.0, p2 - p1)
        z1 = (x1 - mu) / sigma
        z2 = (x2 - mu) / sigma
        desc = f"P({x1:g} ≤ X ≤ {x2:g})"
    elif mode == 'menor' and x1 is not None:
        prob = norm.cdf(x1, mu, sigma)
        z1 = (x1 - mu) / sigma
        z2 = None
        desc = f"P(X ≤ {x1:g})"
    elif mode == 'mayor' and x1 is not None:
        prob = 1.0 - norm.cdf(x1, mu, sigma)
        z1 = (x1 - mu) / sigma
        z2 = None
        desc = f"P(X ≥ {x1:g})"
    else:
        print("\nParámetros insuficientes para el modo seleccionado.")
        return

    # Percentiles
    p50 = norm.ppf(0.5, mu, sigma)
    p90 = norm.ppf(0.9, mu, sigma)
    p95 = norm.ppf(0.95, mu, sigma)

    # Resultados en consola
    print("\n--- Resultados ---")
    print(f"{desc} = {prob:.6f} ({prob:.2%})")
    if z2 is None:
        print(f"z = {z1:.3f}")
    else:
        print(f"z1 = {z1:.3f}, z2 = {z2:.3f}")
    if mode == 'menor':
        print(f"Percentil de {x1:g}: {prob:.2%}")
    elif mode == 'mayor':
        print(f"Porcentaje por encima de {x1:g}: {prob:.2%}")
    else:
        print(f"Masa central entre {x1:g} y {x2:g}: {prob:.2%}")
    print(f"Mediana (p50): {p50:g}, p90: {p90:g}, p95: {p95:g}")

    # Gráficos
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax_pdf, ax_cdf) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # PDF
    x_pdf, y_pdf = shade_region_pdf(ax_pdf, mu, sigma, mode, x1, x2)
    ax_pdf.plot(x_pdf, y_pdf, color='tab:blue', linewidth=2, label='PDF Normal')
    sigma_bands(ax_pdf, mu, sigma, float(np.max(y_pdf)))
    if mode == 'entre' and x1 is not None and x2 is not None:
        ax_pdf.axvline(x1, color='green', linestyle=':', linewidth=1.5)
        ax_pdf.axvline(x2, color='green', linestyle=':', linewidth=1.5)
    else:
        ax_pdf.axvline(float(x1), color='green', linestyle=':', linewidth=1.5)
    ax_pdf.set_ylabel('Densidad', fontsize=12)
    ax_pdf.set_title(f'Distribución Normal (μ={mu:g}, σ={sigma:g})', fontsize=16, pad=10)
    ax_pdf.legend(loc='upper left', fontsize=9, ncols=2)
    ax_pdf.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax_pdf.text(mu, float(np.max(y_pdf)) * 0.7, f"{desc} = {prob:.2%}", fontsize=12,
                ha='center', color='black', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.4'))

    # CDF
    x_cdf, cdf_vals = shade_region_cdf(ax_cdf, mu, sigma, mode, x1, x2)
    ax_cdf.plot(x_cdf, cdf_vals, color='tab:orange', linewidth=2, label='CDF Normal')
    for val, lab in [(p50, 'p50'), (p90, 'p90'), (p95, 'p95')]:
        ax_cdf.axvline(val, color='gray', linestyle='--', linewidth=1)
        ax_cdf.text(val, 0.02, lab, rotation=90, va='bottom', ha='right', fontsize=9, color='gray')
    ax_cdf.set_xlabel('Valores', fontsize=12)
    ax_cdf.set_ylabel('Probabilidad acumulada', fontsize=12)
    ax_cdf.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax_cdf.set_ylim(-0.02, 1.02)
    ax_cdf.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.show()

    # Guardado y duda posterior (aplica siempre: se guarde o no, manual o automático)
    out_path = None
    if ask_save:
        save = input("\n¿Deseas guardar la figura como PNG? (s/n): ").strip().lower()
        if save == 's':
            fname = input("Nombre de archivo (sin extensión, p.ej. 'normal_plot'): ").strip() or 'normal_plot'
            out_dir = get_images_dir()
            out_path = os.path.join(out_dir, f"{fname}.png")
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
            print(f"Figura guardada en {out_path}")
    else:
        if suggested_filename:
            out_dir = get_images_dir()
            out_path = os.path.join(out_dir, f"{suggested_filename}.png")
            fig.savefig(out_path, dpi=200, bbox_inches='tight')
            print(f"Figura guardada en {out_path}")

    # Construir contexto y abrir chat de dudas según flag
    ctx = build_context(mu, sigma, mode, x1, x2, desc, prob,
                        (None if 'z1' not in locals() else z1), (None if 'z2' not in locals() else z2),
                        p50, p90, p95, out_path)
    if start_followup:
        answer_followup_chat_loop(ctx)

def answer_with_chatgpt(context: dict, question: str) -> None:
    """
    Usa ChatGPT (OpenAI) para responder dudas de seguimiento basadas en el contexto del cálculo/gráfica.
    """
    api_key = get_openai_api_key()
    if not api_key:
        print("No hay OPENAI_API_KEY configurada; omitiendo consulta a ChatGPT.")
        return
    system_instructions = (
        "Eres un asistente de estadística que responde en español de forma breve y clara. "
        "Usa EXCLUSIVAMENTE el contexto proporcionado para explicar o aclarar dudas sobre una distribución normal. "
        "Si falta información, indícalo y sugiere cómo obtenerla. Evita inventar datos."
    )
    try:
        try:
            from openai import OpenAI
        except ImportError:
            print("Falta la librería 'openai'. Instálala con: pip install openai")
            return
        preferred_model = os.getenv("OPENAI_MODEL") or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        model_candidates = []
        if preferred_model:
            model_candidates.append(preferred_model)
        model_candidates += [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-3.5-turbo",
        ]
        client = OpenAI(api_key=api_key)
        last_err = None
        content = None
        for m in model_candidates:
            try:
                resp = client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": system_instructions},
                        {"role": "user", "content": f"Contexto: {json.dumps(context, ensure_ascii=False)}"},
                        {"role": "user", "content": f"Pregunta: {question}"},
                    ],
                    temperature=0,
                )
                content = (resp.choices[0].message.content or '').strip()
                print(f"\nRespuesta de ChatGPT ({m}):\n{content}\n")
                break
            except Exception as e:
                last_err = e
                continue
        if content is None:
            print("No se pudo obtener respuesta de ChatGPT.")
            if last_err:
                print(f"Último error: {last_err}")
    except Exception as e:
        print(f"Error consultando ChatGPT: {e}")

def run_manual_mode():
    print("--- Calculadora Avanzada de Distribución Normal (Modo Manual) ---")
    mu = float(input("Introduce la media (μ): "))
    sigma = float(input("Introduce la desviación estándar (σ>0): "))
    if not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
        print("\nError: μ debe ser finito y σ debe ser un número positivo.")
        return

    print("\n¿Qué quieres calcular?")
    print("  1) Probabilidad entre dos valores (x1 < x2)")
    print("  2) Probabilidad menor que un valor (X ≤ x)")
    print("  3) Probabilidad mayor que un valor (X ≥ x)")
    choice = input("Elige 1/2/3: ").strip()

    if choice == '1':
        mode = 'entre'
        x1 = float(input("Introduce x1 (límite inferior): "))
        x2 = float(input("Introduce x2 (límite superior): "))
        if x1 >= x2:
            print("\nError: x1 debe ser menor que x2.")
            return
        plot_and_report(mu, sigma, mode, x1, x2, ask_save=True)
    elif choice == '2':
        mode = 'menor'
        x1 = float(input("Introduce x: "))
        plot_and_report(mu, sigma, mode, x1, None, ask_save=True)
    elif choice == '3':
        mode = 'mayor'
        x1 = float(input("Introduce x: "))
        plot_and_report(mu, sigma, mode, x1, None, ask_save=True)
    else:
        print("\nOpción no válida.")

def run_ai_mode():
    print("--- Calculadora de Distribución Normal con IA (Gemini) ---")
    api_key = get_gemini_api_key()
    if not api_key:
        print("API Key no proporcionada.")
        return

    problem = input(
        "Describe tu problema en lenguaje natural (puedes pegar datos crudos, parámetros deseados, tipo de probabilidad, etc.):\n"
    ).strip()
    if not problem:
        print("No se recibió descripción del problema.")
        return

    # Solicitar que la IA regrese un JSON estructurado
    system_instructions = (
        "Eres un asistente que transforma una solicitud en parámetros para analizar una distribución normal. "
        "Responde EXCLUSIVAMENTE con un JSON válido con los siguientes campos: "
        "{\n"
        "  \"task_type\": \"normal_manual\" | \"fit_data_normal\",\n"
        "  \"mode\": \"entre\" | \"menor\" | \"mayor\",\n"
        "  \"mu\": number | null,\n"
        "  \"sigma\": number | null,\n"
        "  \"x1\": number | null,\n"
        "  \"x2\": number | null,\n"
        "  \"data\": number[] | null,\n"
        "  \"save_png\": boolean | null,\n"
        "  \"filename\": string | null\n"
        "}\n"
        "Reglas: Si el usuario proporciona datos crudos, usa \"fit_data_normal\" y estima mu/sigma desde los datos. "
        "Si no hay datos, usa \"normal_manual\" con mu/sigma proporcionados o inferidos. "
        "Para \"mode\": si pide probabilidad entre dos valores usa \"entre\" y llena x1/x2; si es menor que un valor usa \"menor\" y llena x1; si es mayor usa \"mayor\" y llena x1."
    )

    try:
        try:
            import google.generativeai as genai
        except ImportError:
            print("No se encontró la librería 'google-generativeai'. Instálala con: pip install google-generativeai")
            return

        genai.configure(api_key=api_key)
        prompt = system_instructions + "\n\nSolicitud del usuario:\n" + problem
        # Permitir definir un modelo preferido vía entorno
        preferred_model = os.getenv("GEMINI_MODEL") or os.getenv("GOOGLE_GEMINI_MODEL")
        model_candidates = []
        if preferred_model:
            model_candidates.append(preferred_model)
        # Fallbacks comunes soportados
        model_candidates += [
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash-latest",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ]
        last_err = None
        response = None
        for m in model_candidates:
            try:
                model = genai.GenerativeModel(m)
                response = model.generate_content(prompt)
                print(f"Modelo Gemini usado: {m}")
                break
            except Exception as e:
                last_err = e
                continue
        if response is None:
            print("No se pudo invocar a Gemini con los modelos probados.")
            if last_err:
                print(f"Último error: {last_err}")
            print("Prueba con otro modelo habilitado en tu cuenta o revisa https://ai.google.dev/gemini-api/docs/models")
            return
        text = (response.text or '').strip()
        if not text:
            print("La IA no devolvió contenido.")
            return

        # A veces los modelos envuelven el JSON en bloques de código; intentar limpiar
        if text.startswith("```"):
            # Elimina cercas de código con backticks al inicio/fin
            text = text.strip('`').strip()
        if text.lower().startswith("json"):
            # Quita etiqueta 'json' si aparece
            text = text[4:].strip()

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            print("No se pudo parsear la respuesta de la IA como JSON. Respuesta fue:\n", text)
            return

        task_type = payload.get("task_type")
        mode = payload.get("mode")
        mu = payload.get("mu")
        sigma = payload.get("sigma")
        x1 = payload.get("x1")
        x2 = payload.get("x2")
        data = payload.get("data")
        save_png = bool(payload.get("save_png") or False)
        filename = payload.get("filename") or "normal_plot_ai"

        if task_type == "fit_data_normal":
            if not data or not isinstance(data, list):
                print("La IA solicitó ajuste por datos, pero no proporcionó 'data'.")
                return
            arr = np.array([v for v in data if isinstance(v, (int, float))], dtype=float)
            if arr.size < 2:
                print("Datos insuficientes para estimar μ y σ.")
                return
            mu = float(np.mean(arr))
            sigma = float(np.std(arr, ddof=1)) if arr.size > 1 else float(np.std(arr))

        # Validación final
        if mu is None or sigma is None or not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
            print("Parámetros μ/σ inválidos o faltantes.")
            return
        if mode not in {"entre", "menor", "mayor"}:
            print("Modo inválido o faltante.")
            return
        if mode == 'entre' and (x1 is None or x2 is None or float(x1) >= float(x2)):
            print("x1/x2 inválidos para modo 'entre'.")
            return
        if mode in {"menor", "mayor"} and x1 is None:
            print("x requerido para modo 'menor' o 'mayor'.")
            return

        plot_and_report(float(mu), float(sigma), mode, None if x1 is None else float(x1), None if x2 is None else float(x2),
                        ask_save=(not save_png), suggested_filename=(filename if save_png else None))

    except Exception as e:
        print(f"Error en modo IA: {e}")

def run_chatgpt_mode():
    print("--- Calculadora de Distribución Normal con ChatGPT (OpenAI) ---")
    api_key = get_openai_api_key()
    if not api_key:
        print("API Key de OpenAI no proporcionada.")
        return

    problem = input(
        "Describe tu problema en lenguaje natural (puedes pegar datos crudos, parámetros deseados, tipo de probabilidad, etc.):\n"
    ).strip()
    if not problem:
        print("No se recibió descripción del problema.")
        return

    system_instructions = (
        "Eres un asistente que transforma una solicitud en parámetros para analizar una distribución normal. "
        "Responde EXCLUSIVAMENTE con un JSON válido con los siguientes campos: "
        "{\n"
        "  \"task_type\": \"normal_manual\" | \"fit_data_normal\",\n"
        "  \"mode\": \"entre\" | \"menor\" | \"mayor\",\n"
        "  \"mu\": number | null,\n"
        "  \"sigma\": number | null,\n"
        "  \"x1\": number | null,\n"
        "  \"x2\": number | null,\n"
        "  \"data\": number[] | null,\n"
        "  \"save_png\": boolean | null,\n"
        "  \"filename\": string | null\n"
        "}\n"
        "Reglas: Si el usuario proporciona datos crudos, usa \"fit_data_normal\" y estima mu/sigma desde los datos. "
        "Si no hay datos, usa \"normal_manual\" con mu/sigma proporcionados o inferidos. "
        "Para \"mode\": si pide probabilidad entre dos valores usa \"entre\" y llena x1/x2; si es menor que un valor usa \"menor\" y llena x1; si es mayor usa \"mayor\" y llena x1."
    )

    try:
        try:
            # SDK moderno de OpenAI (>=1.0)
            from openai import OpenAI
        except ImportError:
            print("No se encontró la librería 'openai'. Instálala con: pip install openai")
            return

        preferred_model = os.getenv("OPENAI_MODEL") or os.getenv("AZURE_OPENAI_DEPLOYMENT")
        model_candidates = []
        if preferred_model:
            model_candidates.append(preferred_model)
        model_candidates += [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-3.5-turbo",
        ]

        client = OpenAI(api_key=api_key)
        last_err = None
        content = None
        for m in model_candidates:
            try:
                resp = client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": system_instructions},
                        {"role": "user", "content": problem},
                    ],
                    temperature=0,
                )
                content = (resp.choices[0].message.content or '').strip()
                print(f"Modelo OpenAI usado: {m}")
                break
            except Exception as e:
                last_err = e
                continue
        if content is None:
            print("No se pudo invocar a OpenAI con los modelos probados.")
            if last_err:
                print(f"Último error: {last_err}")
            print("Revisa qué modelos están habilitados en tu plan/gratuito.")
            return

        text = content
        if text.startswith("```"):
            text = text.strip('`').strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            print("No se pudo parsear la respuesta de ChatGPT como JSON. Respuesta fue:\n", text)
            return

        task_type = payload.get("task_type")
        mode = payload.get("mode")
        mu = payload.get("mu")
        sigma = payload.get("sigma")
        x1 = payload.get("x1")
        x2 = payload.get("x2")
        data = payload.get("data")
        save_png = bool(payload.get("save_png") or False)
        filename = payload.get("filename") or "normal_plot_ai"

        if task_type == "fit_data_normal":
            if not data or not isinstance(data, list):
                print("La IA solicitó ajuste por datos, pero no proporcionó 'data'.")
                return
            arr = np.array([v for v in data if isinstance(v, (int, float))], dtype=float)
            if arr.size < 2:
                print("Datos insuficientes para estimar μ y σ.")
                return
            mu = float(np.mean(arr))
            sigma = float(np.std(arr, ddof=1)) if arr.size > 1 else float(np.std(arr))

        if mu is None or sigma is None or not np.isfinite(mu) or not np.isfinite(sigma) or sigma <= 0:
            print("Parámetros μ/σ inválidos o faltantes.")
            return
        if mode not in {"entre", "menor", "mayor"}:
            print("Modo inválido o faltante.")
            return
        if mode == 'entre' and (x1 is None or x2 is None or float(x1) >= float(x2)):
            print("x1/x2 inválidos para modo 'entre'.")
            return
        if mode in {"menor", "mayor"} and x1 is None:
            print("x requerido para modo 'menor' o 'mayor'.")
            return

        plot_and_report(float(mu), float(sigma), mode, None if x1 is None else float(x1), None if x2 is None else float(x2),
                        ask_save=(not save_png), suggested_filename=(filename if save_png else None))

    except Exception as e:
        print(f"Error en modo ChatGPT: {e}")

def main():
    """
    Calcula probabilidades de una Normal(μ, σ) y genera visualizaciones detalladas (PDF y CDF),
    con interpretaciones (z-scores, percentiles) y bandas 68-95-99.7%.
    """
    try:
        print("--- Calculadora de Distribución Normal ---")
        print("Selecciona modo:")
        print("  1) Manual")
        print("  2) IA (Gemini)")
        print("  3) IA (ChatGPT)")
        sel = input("Elige 1/2/3: ").strip()

        if sel == '1':
            run_manual_mode()
        elif sel == '2':
            run_ai_mode()
        elif sel == '3':
            run_chatgpt_mode()
        else:
            print("Opción no válida.")

    except ValueError:
        print("\nError: Por favor, introduce solo valores numéricos.")
    except Exception as e:
        print(f"\nOcurrió un error inesperado: {e}")

if __name__ == "__main__":
    main()
