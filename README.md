# #EsPorAquí – Selección de Hexágonos

Aplicación Streamlit para selección de hexágonos H3 usando métodos de análisis municipal.

## 🚀 Despliegue en Streamlit Cloud

### Opción 1: Desde GitHub (Recomendado)

1. **Sube tu código a GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <tu-repositorio-github>
   git push -u origin main
   ```

2. **Ve a [Streamlit Cloud](https://share.streamlit.io/)**

3. **Haz clic en "New app"**

4. **Conecta tu repositorio de GitHub**

5. **Configura la app:**
   - **Repository**: Selecciona tu repositorio
   - **Branch**: `main` (o la rama que uses)
   - **Main file path**: `app.py`
   - **App URL**: (opcional) Personaliza la URL

6. **Haz clic en "Deploy"**

### Opción 2: Desde Streamlit CLI

```bash
streamlit run app.py
```

## 📋 Requisitos

- Python 3.11+
- Dependencias listadas en `requirements.txt`

## 📁 Estructura del Proyecto

```
esporaqui/
├── app.py                    # Aplicación principal
├── requirements.txt          # Dependencias
├── datum_Sample_data.csv    # Datos de ejemplo
├── .streamlit/
│   └── config.toml          # Configuración de Streamlit
└── README.md                # Este archivo
```

## 🔧 Instalación Local

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En macOS/Linux:
source venv/bin/activate
# En Windows:
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar aplicación
streamlit run app.py
```

## 📝 Notas

- El archivo CSV `datum_Sample_data.csv` debe estar en el directorio raíz del proyecto
- Los usuarios pueden subir su propio CSV desde la interfaz
- La aplicación usa caché de Streamlit para optimizar la carga de datos

