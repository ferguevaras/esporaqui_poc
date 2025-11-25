import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import h3


# ========================================================
# 1. CONVERTIR CATEGORÍAS MUNICIPALES B/M/A/A+ → 1–4
# ========================================================

def convertir_categorias_a_numeros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte categorías municipales B/M/A/A+ a una escala numérica 1–4
    en las columnas:
      - catMunActEcon
      - catMunPob
      - catMunAfluLog
    """
    mapa = {
        "B": 1,
        "M": 2,
        "A": 3,
        "A+": 4
    }

    cols = ["catMunActEcon", "catMunPob", "catMunAfluLog"]

    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map(mapa)

    return df


# ========================================================
# 2. PREFILTRO POR ESTADO Y/O MUNICIPIO
# ========================================================

def prefiltrar_estado_municipio(df: pd.DataFrame, estado: str = None, municipio: str = None) -> pd.DataFrame:
    """
    Devuelve solo los hexágonos del estado y/o municipio seleccionado.
    Si estado o municipio son None, no se filtra por ellos.
    """
    df_f = df.copy()

    if estado is not None:
        df_f = df_f[df_f["noment"].str.lower() == estado.lower()]

    if municipio is not None:
        df_f = df_f[df_f["nomgeo"].str.lower() == municipio.lower()]

    return df_f.reset_index(drop=True)


# ========================================================
# 3. MÉTODO A — FILTRO JERÁRQUICO
# ========================================================

def metodo_A_filtro_jerarquico(
    df: pd.DataFrame,
    min_ae: int = None,
    min_pob: int = None,
    min_afl: int = None
) -> pd.DataFrame:
    """
    Filtrado jerárquico según categorías municipales AE, POB, AFL (1–4):
      - catMunActEcon
      - catMunPob
      - catMunAfluLog

    Mantiene solo los hexágonos que cumplen TODAS las condiciones activas.
    """
    df = df.copy()
    condiciones = []

    if min_ae is not None:
        condiciones.append(df["catMunActEcon"] >= min_ae)

    if min_pob is not None:
        condiciones.append(df["catMunPob"] >= min_pob)

    if min_afl is not None:
        condiciones.append(df["catMunAfluLog"] >= min_afl)

    if condiciones:
        mask = np.logical_and.reduce(condiciones)
        df_filtrado = df.loc[mask]
    else:
        df_filtrado = df

    return df_filtrado.reset_index(drop=True)


# ========================================================
# 4. MÉTODO B — PONDERACIÓN DINÁMICA
# ========================================================

def metodo_B_ponderacion(
    df: pd.DataFrame,
    w_ae: float,
    w_pob: float,
    w_afl: float
) -> pd.DataFrame:
    """
    Calcula score ponderado a nivel municipal:
      score = wAE * catMunActEcon + wPOB * catMunPob + wAFL * catMunAfluLog

    Normaliza el score a 0–100 en la columna 'score_norm'.
    """
    df = df.copy()

    total = w_ae + w_pob + w_afl
    if total == 0:
        # Evitar división por cero: si todo está en 0, asumir pesos iguales.
        w_ae = w_pob = w_afl = 1 / 3
    else:
        w_ae /= total
        w_pob /= total
        w_afl /= total

    df["score"] = (
        df["catMunActEcon"] * w_ae +
        df["catMunPob"] * w_pob +
        df["catMunAfluLog"] * w_afl
    )

    max_score = df["score"].max()
    if max_score and max_score > 0:
        df["score_norm"] = 100 * df["score"] / max_score
    else:
        df["score_norm"] = 0

    return df.sort_values("score_norm", ascending=False).reset_index(drop=True)


# ========================================================
# 5. MÉTODO C — INTERSECCIÓN TOP N MUNICIPAL (CORREGIDO)
# ========================================================

def metodo_C_interseccion(df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """
    Detecta coincidencias en los Top-N rankings municipales de:
      - rankMunActEco
      - rankMunPob
      - rankMunAfluLog

    Requiere columnas:
      - h3_09
      - rankMunActEco
      - rankMunPob
      - rankMunAfluLog
    """
    required = {"h3_09", "rankMunActEco", "rankMunPob", "rankMunAfluLog"}
    if not required.issubset(df.columns):
        raise ValueError(f"Faltan columnas requeridas: {required - set(df.columns)}")

    df = df.copy()

    top_ae = set(df.sort_values("rankMunActEco", ascending=True).head(top_n)["h3_09"])
    top_pob = set(df.sort_values("rankMunPob", ascending=True).head(top_n)["h3_09"])
    top_afl = set(df.sort_values("rankMunAfluLog", ascending=True).head(top_n)["h3_09"])

    registros = []

    for h in df["h3_09"]:
        c = (h in top_ae) + (h in top_pob) + (h in top_afl)
        if c >= 2:
            registros.append({
                "h3_09": h,
                "coincidencias": c,
                "esta_en_AE": h in top_ae,
                "esta_en_POB": h in top_pob,
                "esta_en_AFL": h in top_afl
            })

    out = pd.DataFrame(registros)

    return out.sort_values("coincidencias", ascending=False).reset_index(drop=True)


# ========================================================
# 6. GEO: CONVERSIÓN H3 → POLÍGONO Y MAPA
# ========================================================

def h3_to_polygon(h3_id: str):
    """
    Convierte un hexágono H3 a lista de vértices [lon, lat] usando cell_to_boundary.
    El polígono se cierra automáticamente (el último punto es igual al primero).
    """
    # cell_to_boundary devuelve [(lat, lon), ...] en versiones recientes de h3-py
    boundary = h3.cell_to_boundary(h3_id)
    polygon = [[lon, lat] for lat, lon in boundary]
    # Cerrar el polígono si no está cerrado
    if polygon and polygon[0] != polygon[-1]:
        polygon.append(polygon[0])
    return polygon


def mostrar_hexagonos_en_mapa(df_top3: pd.DataFrame, titulo: str = "Mapa"):
    if df_top3.empty:
        st.warning(f"No hay hexágonos para mostrar en el mapa ({titulo}).")
        return

    # Validar que existe la columna h3_09
    if "h3_09" not in df_top3.columns:
        st.error(f"La columna 'h3_09' no existe en los datos para {titulo}.")
        return

    # Centro del mapa
    h3_id_centro = df_top3.iloc[0]["h3_09"]
    try:
        lat, lon = h3.cell_to_latlng(h3_id_centro)
    except Exception as e:
        st.error(f"Error al obtener coordenadas del hexágono central: {e}")
        return

    # Preparar datos para pydeck
    map_data = []
    for _, row in df_top3.iterrows():
        h3_id = row["h3_09"]
        if pd.isna(h3_id) or not h3_id:
            continue
        try:
            poly = h3_to_polygon(str(h3_id))
            if poly and len(poly) > 0:
                map_data.append({
                    "hex_id": h3_id,
                    "polygon": poly
                })
        except Exception as e:
            st.warning(f"Error al procesar hexágono {h3_id}: {e}")
            continue

    if not map_data:
        st.warning(f"No se pudieron procesar hexágonos válidos para el mapa ({titulo}).")
        return

    # Convertir a DataFrame para pydeck
    df_map = pd.DataFrame(map_data)

    # Crear la capa de polígonos
    polygon_layer = pdk.Layer(
        "PolygonLayer",
        df_map,
        get_polygon="polygon",
        get_fill_color=[255, 0, 0, 120],
        get_line_color=[0, 0, 0],
        line_width_min_pixels=1,
        pickable=True,
        auto_highlight=True,
    )

    # Configurar tooltip para mostrar el ID del hexágono
    tooltip = {
        "html": "<b>ID:</b> {hex_id}",
        "style": {
            "backgroundColor": "steelblue",
            "color": "white",
            "fontSize": "14px",
            "padding": "5px"
        }
    }

    view_state = pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=13,
        pitch=0,
    )

    st.subheader(titulo)
    st.pydeck_chart(pdk.Deck(
        layers=[polygon_layer],
        initial_view_state=view_state,
        tooltip=tooltip
    ))

# ========================================================
# 7. CARGA DE DATOS (CACHE)
# ========================================================

@st.cache_data
def cargar_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = convertir_categorias_a_numeros(df)
    return df


# ========================================================
# 8. APP STREAMLIT
# ========================================================

def main():
    st.set_page_config(
        page_title="#EsPorAquí – Selección de Hexágonos (Municipal)",
        layout="wide",
    )

    st.title("🔷 #EsPorAquí – Selección de Hexágonos (nivel municipal)")
    st.markdown(
        "Prototipo para probar los **métodos A, B y C** usando categorías y rankings municipales."
    )

    # -------------------------
    # Configuración de datos
    # -------------------------
    st.sidebar.header("1. Datos")

    # Ajusta este path a donde tengas tu CSV en tu entorno
    default_path = "datum_Sample_data.csv"
    ruta_manual = st.sidebar.text_input(
        "Ruta del CSV (por defecto)",
        value=default_path,
        help="Ejemplo: /content/drive/.../datum_Sample_data.csv"
    )

    archivo_subido = st.sidebar.file_uploader(
        "O subir un CSV",
        type=["csv"],
        help="Si subes un archivo, se usará en lugar de la ruta."
    )

    if archivo_subido is not None:
        df = pd.read_csv(archivo_subido)
        df = convertir_categorias_a_numeros(df)
    else:
        try:
            df = cargar_dataset(ruta_manual)
        except Exception as e:
            st.error(f"No se pudo cargar el CSV: {e}")
            st.stop()

    st.sidebar.success("✅ Datos cargados.")

    # Validar columnas clave
    columnas_requeridas = [
        "noment", "nomgeo", "h3_09",
        "catMunActEcon", "catMunPob", "catMunAfluLog",
        "rankMunActEco", "rankMunPob", "rankMunAfluLog"
    ]
    faltan = [c for c in columnas_requeridas if c not in df.columns]
    if faltan:
        st.error(f"Faltan columnas requeridas en el dataset: {faltan}")
        st.stop()

    # -------------------------
    # Filtro geográfico
    # -------------------------
    st.sidebar.header("2. Filtro geográfico")

    estados = sorted(df["noment"].dropna().unique().tolist())
    estado_sel = st.sidebar.selectbox(
        "Estado",
        options=["(Todos)"] + estados,
        index=0
    )

    if estado_sel == "(Todos)":
        df_estado = df.copy()
        municipios_opts = sorted(df_estado["nomgeo"].dropna().unique().tolist())
        estado_param = None
    else:
        df_estado = df[df["noment"] == estado_sel]
        municipios_opts = sorted(df_estado["nomgeo"].dropna().unique().tolist())
        estado_param = estado_sel

    municipio_sel = st.sidebar.selectbox(
        "Municipio",
        options=["(Todos)"] + municipios_opts,
        index=0
    )

    if municipio_sel == "(Todos)":
        municipio_param = None
    else:
        municipio_param = municipio_sel

    df_geo = prefiltrar_estado_municipio(
        df,
        estado=estado_param,
        municipio=municipio_param
    )

    st.markdown(f"**Hexágonos tras filtro geográfico:** {len(df_geo):,}")

    if df_geo.empty:
        st.warning("No hay hexágonos para el filtro seleccionado.")
        st.stop()

    # -------------------------
    # Parámetros de métodos
    # -------------------------
    st.sidebar.header("3. Parámetros – Método A (filtro jerárquico)")
    min_ae = st.sidebar.slider("Mínimo AE (catMunActEcon)", 1, 4, 2)
    min_pob = st.sidebar.slider("Mínimo POB (catMunPob)", 1, 4, 2)
    min_afl = st.sidebar.slider("Mínimo AFL (catMunAfluLog)", 1, 4, 2)

    st.sidebar.header("4. Parámetros – Método B (ponderación dinámica)")
    w_ae = st.sidebar.slider("Peso AE", 0.0, 1.0, 0.4, step=0.05)
    w_pob = st.sidebar.slider("Peso POB", 0.0, 1.0, 0.3, step=0.05)
    w_afl = st.sidebar.slider("Peso AFL", 0.0, 1.0, 0.3, step=0.05)

    st.sidebar.header("5. Parámetros – Método C (Top N rankings)")
    top_n = st.sidebar.slider("Top N por variable", 10, 500, 100, step=10)

    # -------------------------
    # Ejecutar algoritmos
    # -------------------------
    if st.button("▶ Ejecutar métodos A, B y C"):
        tabA, tabB, tabC = st.tabs(["Método A", "Método B", "Método C"])

        # ----- Método A -----
        with tabA:
            st.subheader("Método A – Explorador jerárquico (municipal)")
            df_A = metodo_A_filtro_jerarquico(
                df_geo,
                min_ae=min_ae,
                min_pob=min_pob,
                min_afl=min_afl
            )
            st.write(f"Total hexágonos que cumplen filtros: {len(df_A):,}")
            top3_A = df_A.head(3)
            st.write("Top 3 hexágonos (primeros 3 registros):")
            st.dataframe(top3_A)
            mostrar_hexagonos_en_mapa(top3_A, titulo="Mapa – Top 3 Método A")

        # ----- Método B -----
        with tabB:
            st.subheader("Método B – Ponderación dinámica (municipal)")
            df_B = metodo_B_ponderacion(
                df_geo,
                w_ae=w_ae,
                w_pob=w_pob,
                w_afl=w_afl
            )
            st.write(f"Total hexágonos evaluados: {len(df_B):,}")
            top3_B = df_B.head(3)
            st.write("Top 3 hexágonos por score_norm:")
            st.dataframe(top3_B[["h3_09", "score", "score_norm"]])
            mostrar_hexagonos_en_mapa(top3_B, titulo="Mapa – Top 3 Método B")

        # ----- Método C -----
        with tabC:
            st.subheader("Método C – Intersección Top N (municipal)")
            try:
                df_C = metodo_C_interseccion(df_geo, top_n=top_n)
                st.write(f"Total hexágonos con coincidencias ≥ 2: {len(df_C):,}")
                top3_C = df_C.head(3)
                st.write("Top 3 hexágonos por coincidencias:")
                st.dataframe(top3_C)
                mostrar_hexagonos_en_mapa(top3_C, titulo="Mapa – Top 3 Método C")
            except ValueError as e:
                st.error(str(e))


if __name__ == "__main__":
    main()
