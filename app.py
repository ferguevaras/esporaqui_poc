import streamlit as st
import pandas as pd
import numpy as np
import h3
import folium
from streamlit_folium import st_folium


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


def mostrar_hexagonos_en_mapa(df_top10: pd.DataFrame, titulo: str = "Mapa"):
    if df_top10.empty:
        st.warning(f"No hay hexágonos para mostrar en el mapa ({titulo}).")
        return

    # Validar que existe la columna h3_09
    if "h3_09" not in df_top10.columns:
        st.error(f"La columna 'h3_09' no existe en los datos para {titulo}.")
        return

    # Calcular centro del mapa promediando todos los hexágonos
    latitudes = []
    longitudes = []
    hexagonos_validos = []
    
    for _, row in df_top10.iterrows():
        h3_id = row["h3_09"]
        if pd.isna(h3_id) or not h3_id:
            continue
        try:
            lat, lon = h3.cell_to_latlng(str(h3_id))
            latitudes.append(lat)
            longitudes.append(lon)
            hexagonos_validos.append({
                "h3_id": h3_id,
                "lat": lat,
                "lon": lon,
                "row": row
            })
        except Exception as e:
            st.warning(f"Error al obtener coordenadas del hexágono {h3_id}: {e}")
            continue

    if not hexagonos_validos:
        st.warning(f"No se pudieron procesar hexágonos válidos para el mapa ({titulo}).")
        return

    # Calcular centro del mapa
    lat_centro = np.mean(latitudes)
    lon_centro = np.mean(longitudes)

    # Crear mapa de OpenStreetMap que muestra lugares de interés
    m = folium.Map(
        location=[lat_centro, lon_centro],
        zoom_start=13,
        tiles='OpenStreetMap'
    )

    # Agregar cada hexágono al mapa
    for idx, hex_data in enumerate(hexagonos_validos, 1):
        h3_id = hex_data["h3_id"]
        try:
            # Obtener polígono del hexágono
            poly = h3_to_polygon(h3_id)
            
            # Crear polígono de Folium (convertir de [lon, lat] a [lat, lon])
            lat_hex = hex_data["lat"]
            lon_hex = hex_data["lon"]
            folium.Polygon(
                locations=[[coord[1], coord[0]] for coord in poly],  # [lat, lon]
                color='#FF0000',
                weight=2,
                fill=True,
                fillColor='#FF0000',
                fillOpacity=0.3,
                popup=folium.Popup(
                    f"<b>Hexágono #{idx}</b><br><b>ID H3:</b> {h3_id}<br><b>Latitud:</b> {lat_hex:.6f}<br><b>Longitud:</b> {lon_hex:.6f}",
                    max_width=300
                ),
                tooltip=f"Hexágono #{idx}: {h3_id}"
            ).add_to(m)
            
            # Agregar marcador numerado en el centro del hexágono
            folium.CircleMarker(
                location=[lat_hex, lon_hex],
                radius=8,
                popup=folium.Popup(
                    f"<b>Hexágono #{idx}</b><br><b>ID H3:</b> {h3_id}<br><b>Latitud:</b> {lat_hex:.6f}<br><b>Longitud:</b> {lon_hex:.6f}",
                    max_width=300
                ),
                tooltip=f"#{idx} - Lat: {lat_hex:.6f}, Lon: {lon_hex:.6f}",
                color='#000000',
                fill=True,
                fillColor='#FFFFFF',
                fillOpacity=1.0,
                weight=2
            ).add_to(m)
            
            # Agregar número en el centro
            folium.Marker(
                location=[lat_hex, lon_hex],
                icon=folium.DivIcon(
                    html=f'<div style="font-size: 12px; font-weight: bold; color: black; text-align: center; background-color: white; border-radius: 50%; width: 20px; height: 20px; line-height: 20px; border: 2px solid black;">{idx}</div>',
                    icon_size=(20, 20),
                    icon_anchor=(10, 10)
                ),
                tooltip=f"Hexágono #{idx} - Lat: {lat_hex:.6f}, Lon: {lon_hex:.6f}"
            ).add_to(m)
            
        except Exception as e:
            st.warning(f"Error al procesar hexágono {h3_id}: {e}")
            continue

    st.subheader(titulo)
    # Mostrar el mapa en Streamlit - OpenStreetMap ya incluye lugares de interés
    # Usar ancho amplio para aprovechar el espacio horizontal (layout="wide" está configurado)
    st_folium(m, width=1500, height=600, returned_objects=[])

# ========================================================
# 7. CARGA DE DATOS (CACHE)
# ========================================================

@st.cache_data
def cargar_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = convertir_categorias_a_numeros(df)
    return df


# ========================================================
# 8. AUTENTICACIÓN
# ========================================================

def verificar_credenciales(usuario: str, contraseña: str) -> bool:
    """
    Verifica las credenciales del usuario.
    """
    credenciales_validas = {
        "test@efts-group.com": "123prueba"
    }
    return credenciales_validas.get(usuario) == contraseña


def mostrar_pagina_login():
    """
    Muestra la página de inicio de sesión.
    """
    st.set_page_config(
        page_title="Inicio de Sesión – #EsPorAquí",
        layout="centered",
    )
    
    # Centrar el formulario de login
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.title("🔐 Inicio de Sesión")
        st.markdown("---")
        
        with st.form("login_form"):
            usuario = st.text_input("Usuario (Email)", placeholder="usuario@ejemplo.com")
            contraseña = st.text_input("Contraseña", type="password", placeholder="Ingresa tu contraseña")
            submit = st.form_submit_button("Iniciar Sesión", use_container_width=True)
            
            if submit:
                if not usuario or not contraseña:
                    st.error("Por favor, completa todos los campos.")
                elif verificar_credenciales(usuario, contraseña):
                    st.session_state["autenticado"] = True
                    st.session_state["usuario"] = usuario
                    st.success("✅ Inicio de sesión exitoso!")
                    st.rerun()
                else:
                    st.error("❌ Usuario o contraseña incorrectos.")
        
        st.markdown("---")
        st.caption("🔒 Sistema de autenticación - #EsPorAquí")


# ========================================================
# 9. APP STREAMLIT
# ========================================================

def main():
    # Verificar autenticación
    if "autenticado" not in st.session_state:
        st.session_state["autenticado"] = False
    
    if not st.session_state["autenticado"]:
        mostrar_pagina_login()
        return
    
    st.set_page_config(
        page_title="#EsPorAquí – Selección de Hexágonos (Municipal)",
        layout="wide",
    )
    
    # Mostrar información del usuario y botón de cerrar sesión en el sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Usuario:** {st.session_state.get('usuario', 'N/A')}")
    if st.sidebar.button("🚪 Cerrar Sesión", use_container_width=True):
        st.session_state["autenticado"] = False
        st.session_state["usuario"] = None
        st.rerun()
    st.sidebar.markdown("---")

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
    # Documentación de métodos
    # -------------------------
    st.header("📚 Documentación de Métodos")
    
    with st.expander("🔍 Método A: Filtro Jerárquico", expanded=True):
        st.markdown("""
        **¿Cómo funciona?**
        
        El Método A aplica un **filtrado jerárquico** basado en umbrales mínimos para cada categoría municipal:
        - **Actividad Económica (AE)**: Categorías B (1), M (2), A (3), A+ (4)
        - **Población (POB)**: Categorías B (1), M (2), A (3), A+ (4)
        - **Afluencia Logística (AFL)**: Categorías B (1), M (2), A (3), A+ (4)
        
        **Proceso:**
        1. Define umbrales mínimos para cada categoría usando los sliders
        2. Filtra los hexágonos que cumplen **TODAS** las condiciones activas simultáneamente
        3. Retorna todos los hexágonos que pasan el filtro (no hay ranking, solo filtrado)
        
        **Cuándo usarlo:** Cuando necesitas encontrar hexágonos que cumplan criterios mínimos específicos en todas las dimensiones.
        """)
    
    with st.expander("⚖️ Método B: Ponderación Dinámica", expanded=True):
        st.markdown("""
        **¿Cómo funciona?**
        
        El Método B calcula un **score ponderado** combinando las tres categorías municipales con pesos personalizables:
        
        **Fórmula:**
        ```
        score = (wAE × catMunActEcon) + (wPOB × catMunPob) + (wAFL × catMunAfluLog)
        score_norm = (score / max_score) × 100
        ```
        
        **Proceso:**
        1. Asigna pesos a cada categoría (los pesos se normalizan automáticamente)
        2. Calcula el score ponderado para cada hexágono
        3. Normaliza el score a una escala de 0-100
        4. Ordena los hexágonos de mayor a menor score
        
        **Cuándo usarlo:** Cuando quieres priorizar ciertas dimensiones sobre otras y obtener un ranking completo de todos los hexágonos.
        """)
    
    with st.expander("🎯 Método C: Intersección Top N", expanded=True):
        st.markdown("""
        **¿Cómo funciona?**
        
        El Método C identifica hexágonos que aparecen en los **Top N rankings** de múltiples variables simultáneamente:
        - Top N en **Actividad Económica** (rankMunActEco)
        - Top N en **Población** (rankMunPob)
        - Top N en **Afluencia Logística** (rankMunAfluLog)
        
        **Proceso:**
        1. Identifica los Top N hexágonos en cada ranking individual
        2. Encuentra hexágonos que aparecen en **al menos 2 de los 3 rankings**
        3. Cuenta las coincidencias (2 o 3)
        4. Ordena por número de coincidencias (mayor a menor)
        
        **Cuándo usarlo:** Cuando buscas hexágonos que destacan en múltiples dimensiones simultáneamente, identificando áreas con características balanceadas y destacadas.
        """)
    
    st.markdown("---")
    
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
            top10_A = df_A.head(10).copy()
            
            # Agregar latitud y longitud a la tabla
            top10_A["latitud"] = top10_A["h3_09"].apply(lambda x: h3.cell_to_latlng(str(x))[0] if pd.notna(x) else None)
            top10_A["longitud"] = top10_A["h3_09"].apply(lambda x: h3.cell_to_latlng(str(x))[1] if pd.notna(x) else None)
            
            st.write("Top 10 hexágonos (primeros 10 registros):")
            st.dataframe(top10_A)
            
            # Botón de descarga
            csv_A = top10_A.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar Top 10 Método A (CSV)",
                data=csv_A,
                file_name=f"top10_metodo_A_{len(top10_A)}_hexagonos.csv",
                mime="text/csv"
            )
            
            mostrar_hexagonos_en_mapa(top10_A, titulo="Mapa – Top 10 Método A")

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
            top10_B = df_B.head(10).copy()
            
            # Agregar latitud y longitud a la tabla
            top10_B["latitud"] = top10_B["h3_09"].apply(lambda x: h3.cell_to_latlng(str(x))[0] if pd.notna(x) else None)
            top10_B["longitud"] = top10_B["h3_09"].apply(lambda x: h3.cell_to_latlng(str(x))[1] if pd.notna(x) else None)
            
            # Seleccionar columnas para mostrar
            columnas_mostrar = ["h3_09", "latitud", "longitud", "score", "score_norm"]
            st.write("Top 10 hexágonos por score_norm:")
            st.dataframe(top10_B[columnas_mostrar])
            
            # Botón de descarga
            csv_B = top10_B.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Descargar Top 10 Método B (CSV)",
                data=csv_B,
                file_name=f"top10_metodo_B_{len(top10_B)}_hexagonos.csv",
                mime="text/csv"
            )
            
            mostrar_hexagonos_en_mapa(top10_B, titulo="Mapa – Top 10 Método B")

        # ----- Método C -----
        with tabC:
            st.subheader("Método C – Intersección Top N (municipal)")
            try:
                df_C = metodo_C_interseccion(df_geo, top_n=top_n)
                st.write(f"Total hexágonos con coincidencias ≥ 2: {len(df_C):,}")
                top10_C = df_C.head(10).copy()
                
                # Agregar latitud y longitud a la tabla
                top10_C["latitud"] = top10_C["h3_09"].apply(lambda x: h3.cell_to_latlng(str(x))[0] if pd.notna(x) else None)
                top10_C["longitud"] = top10_C["h3_09"].apply(lambda x: h3.cell_to_latlng(str(x))[1] if pd.notna(x) else None)
                
                st.write("Top 10 hexágonos por coincidencias:")
                st.dataframe(top10_C)
                
                # Botón de descarga
                csv_C = top10_C.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Descargar Top 10 Método C (CSV)",
                    data=csv_C,
                    file_name=f"top10_metodo_C_{len(top10_C)}_hexagonos.csv",
                    mime="text/csv"
                )
                
                mostrar_hexagonos_en_mapa(top10_C, titulo="Mapa – Top 10 Método C")
            except ValueError as e:
                st.error(str(e))


if __name__ == "__main__":
    main()
