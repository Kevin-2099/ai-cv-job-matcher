# 🤖 AI CV Job Matcher
Aplicación de IA que analiza CVs contra ofertas de trabajo para calcular compatibilidad multidimensional, detectar brechas técnicas, evaluar la calidad del CV y generar informes exportables. Diseñada tanto para candidatos como para reclutadores.

## 🚀 Funcionalidades
- 👤 Modo Candidato

  - 📄 Sube tu CV en PDF o pega el texto directamente (o combina ambos)
  - 📋 Sube la oferta en PDF o pégala como texto
  - 📊 Análisis individual con puntuación global y por dimensiones
  - 🏆 Ranking multi-oferta: compara tu CV contra varias ofertas a la vez

- 🏢 Modo Recruiter

  - 📂 Sube múltiples CVs en PDF o pégalos como bloques de texto
  - 🔍 Analiza todos los candidatos contra una misma oferta
  - 📊 Tabla de ranking con puntuaciones detalladas por candidato
  - 📋 Informes individuales expandibles para cada CV

- 📐 Puntuación multidimensional
  - El score se calcula en 5 dimensiones independientes:
    - ⚙️ Técnica 40%
    - 💬 Habilidades blandas 20%
    - 🗓️ Experiencia 15%
    - 🌐 Idiomas 15%
    - 🎓 Formación 10%
  - Visualizado en un gráfico de radar interactivo.

- 🧠 Análisis inteligente

  - 🏷️ Detección automática de sector desde la oferta de trabajo
  - 🔑 Extracción de keywords filtrada por sector para mayor precisión
  - ⚠️ Brechas ponderadas por frecuencia en la oferta
  - 💪 Fortalezas detectadas y resaltadas visualmente
  - 🌐 Detección de idiomas requeridos vs. disponibles en el CV
  - 🗓️ Extracción de años de experiencia requeridos vs. aportados
  - 🎓 Comparación del nivel de formación académica

- ✍️ Calidad del CV

  - Detección de verbos débiles con sugerencias de reemplazo (ej: "participé en" → "lideré")
  - Identificación de secciones faltantes (contacto, experiencia, formación, habilidades)
  - Densidad de keywords: cuántas veces aparece cada keyword de la oferta en tu CV
  - Vista previa del CV con keywords resaltadas en verde (presentes) y rojo (ausentes)

- 📥 Exportación e historial

  - Exporta el informe completo como PDF con un solo clic
  - Historial de los últimos 10 análisis guardado en sesión

## 🛠️ Tecnologías

- Streamlit
- Sentence Transformers — all-MiniLM-L6-v2
- Scikit-learn — similitud coseno
- PyPDF2 — extracción de texto PDF
- Plotly — gráfico de radar
- fpdf2 — exportación PDF
- Pandas — tablas de ranking

## ▶️ Cómo usar

1️⃣ Clona este repositorio

git clone https://github.com/Kevin-2099/ai-cv-job-matcher

2️⃣ Instala dependencias

pip install -r requirements.txt

3️⃣ Ejecuta la aplicación

streamlit run app.py

4️⃣ Abre en el navegador el link que Streamlit indique

## 💡 Notas

La primera ejecución puede tardar unos segundos mientras se descarga el modelo de IA.

Compatible con CPU (no requiere GPU).

Puedes combinar PDF subido + texto pegado simultáneamente en cualquier campo.

En el modo multi-oferta o multi-CV, separa los bloques de texto con ---.

## 📄 Licencia

Este proyecto se distribuye bajo una **licencia propietaria con acceso al código (source-available)**.

El código fuente se pone a disposición únicamente para fines de **visualización, evaluación y aprendizaje**.

❌ No está permitido copiar, modificar, redistribuir, sublicenciar, ni crear obras derivadas del software o de su código fuente sin autorización escrita expresa del titular de los derechos.

❌ El uso comercial del software, incluyendo su oferta como servicio (SaaS), su integración en productos comerciales o su uso en entornos de producción, requiere un **acuerdo de licencia comercial independiente**.

📌 El texto **legalmente vinculante** de la licencia es la versión en inglés incluida en el archivo `LICENSE`. 

Se proporciona una traducción al español en `LICENSE_ES.md` únicamente con fines informativos. En caso de discrepancia, prevalece la versión en inglés.

## Autor
Kevin-2099
