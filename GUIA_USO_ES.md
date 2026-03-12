# Guia de Uso en Espanol

Esta guia resume como usar el proyecto en los dos escenarios principales:

- uso local en Spyder o un IDE parecido
- uso en Google Colab

## Elegir el entorno

| Donde lo vas a correr | Archivo principal | Que hacer |
|---|---|---|
| Google Colab | [GR_python_colab/GR_Colab.ipynb](C:/Users/Nelson/Downloads/GR_python/GR_python_colab/GR_Colab.ipynb) | Ejecutar las celdas en orden |
| Spyder o Python local | [gr_main.py](C:/Users/Nelson/Downloads/GR_python/gr_main.py) | Elegir `METRIC_KEY` y correr [gr_calculator.py](C:/Users/Nelson/Downloads/GR_python/gr_calculator.py) |
| Google Cloud o Linux remoto | [gr_main.py](C:/Users/Nelson/Downloads/GR_python/gr_main.py) | Instalar dependencias y correr [gr_calculator.py](C:/Users/Nelson/Downloads/GR_python/gr_calculator.py) |

## 1. Uso local paso a paso

### Preparacion inicial

1. Abri la carpeta del proyecto en Spyder.
2. Verifica que Spyder este usando un entorno de Python que tenga `sympy` instalado.
3. Si queres generar el PDF directamente, instala una distribucion LaTeX y asegura que `pdflatex` este en PATH.

### Correr una metrica built-in

1. Abri [gr_main.py](C:/Users/Nelson/Downloads/GR_python/gr_main.py).
2. En la Seccion 1, elegi una metrica con una sola linea. Por ejemplo:
   ```python
   METRIC_KEY = 'schwarzschild'
   ```
3. Corre [gr_calculator.py](C:/Users/Nelson/Downloads/GR_python/gr_calculator.py) o [gr_main.py](C:/Users/Nelson/Downloads/GR_python/gr_main.py).
4. Revisa los archivos generados en la carpeta del proyecto:
   - `gr_report.tex`
   - `gr_report.pdf` si `pdflatex` esta disponible

### Correr una metrica custom

1. Abri [gr_main.py](C:/Users/Nelson/Downloads/GR_python/gr_main.py).
2. Cambia a:
   ```python
   METRIC_KEY = 'custom'
   ```
3. En la Seccion 1.2, define los simbolos o funciones extra que necesites.
4. Completa `CUSTOM_METRIC_CONFIG`.
5. Corre el script.

Ejemplos ya incluidos en [gr_main.py](C:/Users/Nelson/Downloads/GR_python/gr_main.py):

- una metrica diagonal con `alpha(r)`
- una metrica con termino cruzado `dt dr` usando `beta(r)`

## 2. Uso en Colab paso a paso

1. Abri [GR_python_colab/GR_Colab.ipynb](C:/Users/Nelson/Downloads/GR_python/GR_python_colab/GR_Colab.ipynb) en Google Colab.
2. Corre **Cell 1** para instalar dependencias.
3. Corre **Cell 2** para clonar el repo e importar los modulos.
4. Edita **Cell 3**.
5. Corre **Cell 4** para el calculo simbolico.
6. Corre **Cell 5** para generar los reportes.
7. Si queres numerica y graficos, segui con **Cell 6**, **Cell 7** y **Cell 8**.

### Configuracion tipica en Colab

En **Cell 3**, la configuracion usual del flujo warp-document es:

```python
VARIANT = 'variant_a'
PROFILE_MODE = 'document_generic'
RUN_DOCUMENT_COMPARISON = True
GENERATE_COMPARISON_REPORT = True
```

Eso produce:

- `gr_report.pdf` como salida directa de la corrida simbolica
- `gr_comparison_report.pdf` como comparacion opcional contra formulas externas

## 3. Metricas built-in disponibles

Las metricas registradas actualmente en [gr_metric_library.py](C:/Users/Nelson/Downloads/GR_python/gr_metric_library.py) son:

- `schwarzschild`
- `reissner_nordstrom`
- `de_sitter_static`
- `minkowski_spherical`
- `frw_flat`
- `static_spherical`
- `morris_thorne_wormhole`
- `pg_areal`
- `pg_spatial_conformal`
- `warp_doc_baseline`
- `warp_doc_variant_a`
- `warp_doc_variant_b`

Las tres opciones `warp_doc_baseline`, `warp_doc_variant_a` y `warp_doc_variant_b` corresponden directamente a las metricas del documento de ustedes.

## 4. Como agregar una metrica nueva

### Opcion A: metrica custom para una sola corrida

1. Usa `METRIC_KEY = 'custom'`.
2. Define en la Seccion 1.2 las funciones o parametros extra que necesites.
3. Completa `CUSTOM_METRIC_CONFIG`.
4. Corre el proyecto.

### Opcion B: metrica built-in permanente

1. Abri [gr_metric_library.py](C:/Users/Nelson/Downloads/GR_python/gr_metric_library.py).
2. Agrega una nueva entrada dentro de `build_builtin_metric_library()`.
3. Usa las mismas claves:
   - `g_metric`
   - `metric_name`
   - `metric_description`
   - opcional `g_inv_metric`
   - opcional `e_tetrad`
4. Luego selecciona esa nueva metrica desde [gr_main.py](C:/Users/Nelson/Downloads/GR_python/gr_main.py) con:
   ```python
   METRIC_KEY = 'tu_nueva_metrica'
   ```
