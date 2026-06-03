# Spec — Función de extracción de frames de un vídeo

- **Tarea atómica:** `frame_extraction`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni
  modificar código aún.

---

## 1. Requisito (historia de usuario)

> **Como** persona que desarrolla el pipeline de análisis de fútbol robótico,
> **quiero** una función que extraiga frames de un vídeo, ya sea una **cuota fija**
> de frames distribuidos uniformemente en el tiempo o **todos** los frames
> disponibles,
> **para** disponer de muestras de imagen consistentes y configurables que
> alimenten las etapas posteriores de detección, segmentación y rastreo.

---

## 2. Motivación (por qué)

- Las etapas del pipeline (YOLO → SAM3 → ByteTrack) operan sobre **imágenes**, no
  sobre vídeos; se necesita una pieza base que convierta un vídeo en frames.
- Procesar **todos** los frames de 123 vídeos es costoso. Extraer una **cuota**
  representativa, repartida uniformemente a lo largo del vídeo, permite muestrear
  el partido completo a un coste controlado.
- Mantener la cuota en el **archivo de configuración** (no en el código) habilita
  experimentar con distintos valores por *trial* sin tocar el código, conforme a
  la constitución.

---

## 3. Alcance

### 3.1 Dentro de alcance

- Definir una **función de extracción de frames** a partir de la ruta de un
  **único vídeo**.
- Soportar **dos modos** de extracción:
  - **Modo cuota** (por defecto): extraer una cantidad fija de frames repartidos
    de forma **uniforme en el tiempo** a lo largo del vídeo.
  - **Modo completo:** extraer **todos** los frames disponibles del vídeo.
- Leer el valor de la **cuota** desde el archivo de configuración `.json` del
  proyecto.
- **Verificar que la ruta del vídeo sea real** reutilizando la utilidad existente
  del proyecto.

### 3.2 Fuera de alcance

- **Escribir los frames a disco** (guardado, formato de salida, nomenclatura de
  archivos): esta tarea entrega los frames como resultado; la persistencia se
  abordará por separado.
- Procesar **directorios o lotes** de vídeos (la función opera sobre un vídeo).
- Cualquier lógica de **detección, segmentación, tracking** o preprocesamiento
  posterior de los frames.
- La definición del **cómo técnico** (librería de lectura de vídeo, tipos exactos,
  ubicación del módulo, mecanismo de lectura de la configuración): corresponde al
  `plan.md`.

---

## 4. Comportamiento esperado

- **Entrada:** la **ruta de un vídeo** y un indicador del modo de extracción
  (cuota vs. todos los frames).
- **Salida:** los **frames extraídos**, entregados como resultado de la función
  (en memoria); esta tarea **no** los escribe a disco.
- **Modo por defecto:** **cuota** (extracción parcial). El modo completo se activa
  de forma explícita.
- **Modo cuota:**
  - La cantidad de frames objetivo se lee desde el **archivo de configuración**
    `.json` (la constitución exige que estos parámetros vivan en configuración).
    El valor `30` del borrador es solo un **ejemplo**.
  - Los frames se reparten **uniformemente a lo largo de la duración** del vídeo
    (equiespaciados en el tiempo), no los primeros N ni una selección aleatoria.
    La función calcula, según la duración/cantidad de frames del vídeo, cada
    cuánto debe tomar un frame para llegar a la cuota.
  - Si el vídeo tiene **menos frames que la cuota** solicitada, la cuota se trata
    como un **máximo**: se devuelven todos los frames disponibles sin duplicar ni
    rellenar.
- **Modo completo:** se devuelven **todos** los frames disponibles del vídeo.
- **Validación de la ruta:** la función comprueba que la ruta recibida sea real
  **reutilizando `src/utils.py::get_abs_path`**, que resuelve la ruta respecto a
  la raíz del proyecto y **lanza** `FileNotFoundError` (ruta inexistente) o
  `ValueError` (entrada inválida). La función se apoya en ese comportamiento para
  detener el proceso ante una ruta no válida.

---

## 5. Criterios de aceptación

1. **AC-1 — Función presente:** existe una función de extracción de frames para
   un vídeo.
2. **AC-2 — Dos modos:** la función soporta el modo **cuota** (por defecto) y el
   modo **todos los frames**, seleccionables mediante un indicador de entrada.
3. **AC-3 — Cuota desde configuración:** el número de frames del modo cuota se
   obtiene del archivo de configuración `.json` del proyecto, **no** del código.
4. **AC-4 — Distribución uniforme:** en modo cuota, los frames devueltos están
   repartidos **uniformemente en el tiempo** a lo largo del vídeo y su cantidad
   coincide con la cuota (o con el total disponible si el vídeo tiene menos
   frames que la cuota).
5. **AC-5 — Modo completo:** en modo completo, la función devuelve todos los
   frames disponibles del vídeo.
6. **AC-6 — Ruta verificada:** la función verifica la realidad de la ruta del
   vídeo reutilizando `get_abs_path`; ante una ruta inexistente o inválida, el
   proceso se detiene con el error correspondiente
   (`FileNotFoundError` / `ValueError`).
7. **AC-7 — Salida en memoria:** la función entrega los frames extraídos como
   resultado, sin escribirlos a disco (eso queda fuera de alcance).
8. **AC-8 — Validación manual:** se demuestra de forma **exploratoria** (script
   suelto o notebook) que, sobre un vídeo real, el modo cuota devuelve la cantidad
   esperada de frames repartidos en el tiempo y el modo completo devuelve todos
   los frames.

---

## 6. Supuestos y notas

- La cuota por defecto del borrador (`30`) es un **ejemplo**; el valor efectivo
  siempre proviene de la configuración.
- Recordatorio de la constitución y del proyecto: los vídeos solo resuelven
  **dentro del contenedor** (los datos se montan ahí); en el host las rutas de
  datos pueden ser enlaces muertos. Por tanto, la **validación manual** que toque
  vídeos reales debe ejecutarse en el contenedor.
- Esta especificación **no** define el *cómo* técnico (librería de lectura de
  vídeo, tipos de los parámetros, módulo destino dentro de `src/`, formato exacto
  de la salida ni la clave de configuración de la cuota); todo ello corresponde al
  `plan.md` de esta misma carpeta.

---

## 7. Siguientes pasos (metodología)

1. Elaborar `plan.md` con el detalle técnico de implementación
   (apoyándose en el borrador `.specs/drafts/frame_extraction/00_plan.md`).
2. Derivar `tasks.md` con las tareas ejecutables.
3. Implementar (paso 5) únicamente después de los anteriores.
