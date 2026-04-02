# Hugging Face y el modelo Cosmos 2B — qué es y cómo lo usa el equipo

## ¿Qué es Hugging Face?

**Hugging Face** ([huggingface.co](https://huggingface.co)) es un sitio donde publican **modelos de IA** (pesos + configuración). Tu código no “está dentro” de Hugging Face: **descarga** el modelo a tu PC la primera vez que ejecutas algo como `from_pretrained("nvidia/Cosmos-Reason2-2B")`.

- **No es** un servidor donde corre tu app.
- **Sí es** el catálogo + almacén desde el que **Python descarga** archivos si tienes internet (o usas una carpeta local ya descargada).

## Cómo lo especifica el proyecto (tus teammates)

En el código, el id del modelo viene de:

1. Variable de entorno **`COSMOS_MODEL`** (en `.env`), por defecto `nvidia/Cosmos-Reason2-2B`, o  
2. El código en `model_handler.py` (`DEFAULT_COSMOS_MODEL`).

La primera vez que alguien con **red buena** ejecuta la app, **Transformers** descarga el modelo al **caché local** (normalmente bajo tu usuario, carpeta tipo `.cache/huggingface`). Las siguientes veces **no hace falta** internet si ya está en caché.

### Comandos útiles

```powershell
# Ver si estás logueado (modelos privados o gated)
hf auth whoami

# Descargar el repo a una carpeta (otra red / USB)
hf download nvidia/Cosmos-Reason2-2B --local-dir .\models\Cosmos-Reason2-2B
```

Luego en `.env`:

```env
COSMOS_MODEL=C:/ruta/completa/a/models/Cosmos-Reason2-2B
MOCK_COSMOS=0
```

### Si no puedes descargar

- **`MOCK_COSMOS=1`** en `.env`: no usa Cosmos; captions falsos para probar UI y resúmenes.
- Copiar la carpeta del modelo desde un compañero que ya lo tenga descargado.

## Gated models

Algunos modelos piden **aceptar la licencia** en la web del repo con la misma cuenta que usas en `hf auth login`.

### Error 403 al descargar (`Cosmos-Reason2-2B`)

Ese modelo es **gated**. Además de aceptar en la página del modelo, tu **token** debe poder leer repos gated:

1. Abre [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
2. Si usas un token **fine-grained**: edítalo y activa **“Access to public gated repositories”** (o crea un token **classic** con rol *Read*).
3. Vuelve a loguearte:

   ```powershell
   .\.venv\Scripts\hf.exe auth login --token TU_TOKEN
   ```

4. Descarga (puede tardar y ocupar varios GB):

   ```powershell
   cd "ruta\a\Nvidia_COSMOS"
   .\.venv\Scripts\hf.exe download nvidia/Cosmos-Reason2-2B --local-dir .\models\Cosmos-Reason2-2B
   ```

5. En `.env`:

   ```env
   MOCK_COSMOS=0
   COSMOS_MODEL=C:/ruta/completa/Nvidia_COSMOS/models/Cosmos-Reason2-2B
   ```

   (Ajusta la ruta; usa barras `/` o escapa `\` en Windows.)
