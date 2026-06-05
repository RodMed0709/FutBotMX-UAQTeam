# Plan técnico — Videos fijados al split de testing (`forced_testing_split`)

- **Tarea atómica:** `forced_testing_split`
- **Paso de la metodología:** 3 (Planificación técnica)
- **Spec de referencia:** [`spec.md`](./spec.md)
- **Estado:** Diseño técnico. **No** implica crear ni modificar código aún.

---

## 1. Objetivo del plan

Definir cómo extender `src/data/metadata.py` para que una lista de videos
configurada quede siempre en testing (`split = 2`), repartiendo el resto al azar con
la seed existente, sin alterar conteos ni reproducibilidad.

---

## 2. Stack técnico

- Sin dependencias nuevas. Reutiliza `decord`, `pandas`, `numpy` ya en uso.

---

## 3. Diseño

### 3.1 Config (`configs/00_testing_config.json`)

Nueva sección `splits` con la lista de rutas fijadas a testing (opcional):

```jsonc
"splits": {
  "forced_testing": [
    "data/raw/18abril/Camara_superior/IMG_9933.MOV",
    "data/raw/18abril/Camara_superior/IMG_9938.MOV"
  ]
}
```

Si la clave o la sección no existen, se asume lista vacía (compatibilidad con la
tarea previa).

### 3.2 `_load_metadata_config`

Pasa a devolver **4** valores: `(dataset_dir, metadata_csv, split_seed,
forced_testing)`, donde `forced_testing` es `list[str]` (vacía si ausente). El resto
del parseo no cambia.

### 3.3 `_assign_splits` (nueva firma)

```python
def _assign_splits(n: int, seed: int, forced_testing_idx: list[int]) -> list[int]:
    forced = set(forced_testing_idx)
    n_testing = SPLIT_SIZES[SPLIT_TESTING]
    if len(forced) > n_testing:
        raise ValueError(...)                      # más fijados que plazas de testing
    if n < sum(SPLIT_SIZES.values()):
        raise ValueError(...)                      # videos insuficientes

    splits = [SPLIT_RESERVE] * n
    for idx in forced:
        splits[idx] = SPLIT_TESTING

    pool = [i for i in range(n) if i not in forced] # candidatos al reparto aleatorio
    rng = np.random.default_rng(seed)
    shuffled = [pool[i] for i in rng.permutation(len(pool))]

    cursor = 0
    remaining_testing = n_testing - len(forced)
    for idx in shuffled[cursor:cursor + remaining_testing]:
        splits[idx] = SPLIT_TESTING
    cursor += remaining_testing
    for idx in shuffled[cursor:cursor + SPLIT_SIZES[SPLIT_FINETUNING]]:
        splits[idx] = SPLIT_FINETUNING
    return splits
```

- Fijados disjuntos del pool por construcción; conteos: testing = `len(forced) +
  remaining_testing = 20`, fine-tuning = 23, reserva = resto.
- Reproducibilidad intacta: la aleatoriedad solo actúa sobre `pool` con la misma seed.

### 3.4 `build_metadata_csv`

- Tras `_discover_videos`, construir el mapa `ruta -> idx`.
- Resolver `forced_testing_idx` desde las rutas configuradas; si alguna ruta **no**
  está entre los videos descubiertos → `ValueError` claro.
- Pasar `forced_testing_idx` a `_assign_splits`. El resto del flujo no cambia.

### 3.5 Manejo de errores

| Situación | Excepción |
|---|---|
| `> 20` rutas fijadas | `ValueError` |
| Ruta fijada inexistente en el dataset | `ValueError` |
| (heredados de la tarea previa) | sin cambios |

---

## 4. Cambios de configuración

- Añadir `splits.forced_testing` (ver §3.1). `seeds.split` y `working_dirs` no
  cambian.

---

## 5. Validación

`testing/test_metadata.py` (agente, local):

- Actualizar el desempaquetado de `_load_metadata_config` a 4 valores.
- Añadir comprobación: todos los videos de `splits.forced_testing` tienen
  `split == 2`.
- Mantener las comprobaciones previas (conteos 23/20/resto, reproducibilidad,
  idempotencia, handler).
- `ruff`/`black` limpios.

---

## 6. Trazabilidad

| Criterio (spec) | Cubierto por |
|---|---|
| AC-1 Config | §3.1, §3.2 |
| AC-2 Fijados en testing | §3.3, §3.4 |
| AC-3 Conteos | §3.3 |
| AC-4 Resto reproducible | §3.3 |
| AC-5 Lista vacía | §3.1, §3.3 |
| AC-6 Errores | §3.3, §3.4, §3.5 |
| AC-7 CSV regenerado | §5 (borrar + `force=True`) |
| AC-8 Validación local | §5 |

---

## 7. Riesgos y consideraciones

- **Estabilidad de índices:** los `forced_testing_idx` se resuelven cada vez desde la
  `ruta` (no del `id`), así que añadir/quitar videos no rompe la fijación mientras la
  ruta exista.
- **Compatibilidad:** con `forced_testing` vacío el resultado es idéntico al de
  `csv_dataset_metadata` (la permutación cubre todo el conjunto).
