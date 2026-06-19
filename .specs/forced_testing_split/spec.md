# Spec — Videos fijados al split de testing (`forced_testing_split`)

- **Tarea atómica:** `forced_testing_split`
- **Paso de la metodología:** 2 (Especificación)
- **Estado:** Definición del *qué* y el *porqué*. **No** implica crear ni
  modificar código aún.

---

## 1. Requisito (historia de usuario)

> **Como** persona que prepara el dataset de fútbol robótico,
> **quiero** que ciertos videos concretos queden **siempre** en el split de
> *testing* al generar `db_metadata.csv`, mientras el resto se reparte de forma
> aleatoria reproducible,
> **para** garantizar que escenarios clave (p. ej. los dos videos de
> `data/raw/18abril/Camara_superior`) se evalúen siempre en testing y no dependan
> del azar de la seed.

---

## 2. Motivación (por qué)

- La tarea `csv_dataset_metadata` asigna los splits **100 % al azar** (con seed).
  No hay forma de garantizar que un video específico caiga en *testing*.
- Hoy eso se está parchando **a mano** sobre el CSV (intercambio manual de splits),
  cambio que se pierde en cuanto se regenera con `force=True`. Es frágil y no
  reproducible.
- Se necesita que esa fijación viva en la **configuración** y la respete el
  generador, manteniendo el resto del reparto aleatorio y los conteos por split.

---

## 3. Alcance

### 3.1 Dentro de alcance

- Añadir a la config global una lista de **videos fijados a testing**
  (rutas relativas a `PROJECT_ROOT`).
- Modificar la construcción del manifiesto (`src/data/metadata.py`) para que esos
  videos reciban siempre `split = 2` (testing), y el resto de las plazas de testing
  + las de fine-tuning se asignen **aleatoriamente** con la seed existente.
- Mantener **invariantes**: conteos por split sin cambio (23 / 20 / resto),
  reproducibilidad del reparto aleatorio, splits disjuntos y cubrientes.
- **Regenerar** `assets/db_metadata.csv`: borrar el existente y crear el nuevo con
  la nueva lógica.
- Actualizar la validación local (`testing/test_metadata.py`).

### 3.2 Fuera de alcance

- Fijar videos a otros splits (solo testing en esta tarea).
- Cualquier cambio al pipeline o a `extract_frames`.
- El **cómo técnico** (firma exacta, formato de config): corresponde al `plan.md`.

---

## 4. Comportamiento esperado

- **Config:** una lista (posiblemente vacía) de rutas relativas; si está vacía o
  ausente, el comportamiento es idéntico al de `csv_dataset_metadata` (todo al azar).
- **Generación:**
  - Los videos cuya `ruta` esté en la lista → `split = 2` siempre.
  - Las plazas restantes de testing (`20 − nº fijados`) y las de fine-tuning (23) se
    asignan al azar (seed) entre los **no fijados**; el resto es reserva (0).
  - Si la lista tiene más de 20 entradas, o referencia un video inexistente en el
    dataset → error claro.
- **Conteos finales:** testing = 20, fine-tuning = 23, reserva = resto.

---

## 5. Criterios de aceptación

1. **AC-1 — Config:** existe en la config una lista de videos fijados a testing,
   leída por el generador (sin hardcodear).
2. **AC-2 — Fijados en testing:** todos los videos de la lista tienen `split = 2`
   en el CSV generado.
3. **AC-3 — Conteos:** testing = 20, fine-tuning = 23, reserva = resto; disjuntos y
   cubrientes.
4. **AC-4 — Resto aleatorio reproducible:** con la misma seed, el reparto de los no
   fijados es idéntico entre corridas.
5. **AC-5 — Lista vacía:** con lista vacía/ausente, el resultado equivale al de
   `csv_dataset_metadata`.
6. **AC-6 — Errores:** lista con > 20 entradas o ruta inexistente → error claro.
7. **AC-7 — CSV regenerado:** el `assets/db_metadata.csv` previo se borra y se crea
   uno nuevo con la lógica de fijación.
8. **AC-8 — Validación local:** `testing/test_metadata.py` verifica AC-2..AC-4 y
   pasa en local.

---

## 6. Supuestos y notas

- Reutiliza toda la maquinaria de `csv_dataset_metadata` (descubrimiento, extracción
  con decord, escritura/validación del CSV). Solo cambia la **asignación de splits**.
- Las rutas fijadas se comparan contra la columna `ruta` (relativa a `PROJECT_ROOT`,
  POSIX).
- La seed sigue siendo única (`seeds.split`); la fijación es determinista y la
  aleatoriedad del resto, reproducible.
