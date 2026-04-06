import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import re
import plotly.graph_objects as go
import pandas as pd
from fpdf import FPDF
import hashlib
import datetime

# ─────────────────────────────────────────────
# LANGUAGE STRINGS
# ─────────────────────────────────────────────
LANGUAGES = {
    "English": {
        "mode_select": "Select mode",
        "mode_candidate": "👤 Candidate",
        "mode_recruiter": "🏢 Recruiter",
        "upload_cv": "📄 Upload CV (PDF)",
        "upload_cvs": "📄 Upload CVs (multiple PDFs)",
        "upload_job": "📄 Upload Job Description (PDF)",
        "paste_cv": "Or paste CV text",
        "paste_job": "📋 Paste job description (or upload PDF above)",
        "paste_jobs": "📋 Paste job descriptions (separate with ---)",
        "analyze": "Analyze",
        "match_result": "📊 Match Result",
        "section_scores": "📐 Score by Section",
        "strengths_detected": "💪 Strengths Detected",
        "gaps_vs_job": "⚠️ Gaps vs Job Offer",
        "gap_importance": "Gap Importance",
        "recommendations": "🧠 Recommendations",
        "no_cv_job": "Please provide a CV and a job description.",
        "none_detected": "None detected",
        "highly_aligned": "Your profile is highly aligned! Apply with confidence 🎯",
        "consider_learning": "Consider learning or reinforcing: ",
        "improve_cv": "Rewrite your CV to better highlight relevant experience.",
        "analysis_done": "Analysis completed 🚀",
        "weak_verbs": "✍️ Weak Verb Suggestions",
        "missing_sections": "📋 Missing CV Sections",
        "keyword_density": "🔑 Keyword Density",
        "cv_in_job": "CV", "job_in_job": "Job",
        "generic_cv_warning": "⚠️ Your CV seems generic — consider tailoring it for this specific role.",
        "experience_match": "🗓️ Experience",
        "language_match": "🌐 Languages",
        "education_match": "🎓 Education",
        "cv_preview": "📄 CV Keywords Highlighted",
        "history": "🕓 Analysis History",
        "no_history": "No previous analyses yet.",
        "export_pdf": "📥 Export Report as PDF",
        "multi_job_ranking": "📊 Job Ranking",
        "multi_cv_ranking": "📊 CV Ranking",
        "sector_detected": "🏷️ Detected Sector",
        "years_cv": "Years in CV",
        "years_job": "Years required",
        "languages_cv": "Languages in CV",
        "languages_job": "Languages in Job",
        "degree_cv": "Degree in CV",
        "degree_job": "Degree in Job",
    },
    "Español": {
        "mode_select": "Selecciona el modo",
        "mode_candidate": "👤 Candidato",
        "mode_recruiter": "🏢 Recruiter",
        "upload_cv": "📄 Subir CV (PDF)",
        "upload_cvs": "📄 Subir CVs (múltiples PDFs)",
        "upload_job": "📄 Subir Oferta de Trabajo (PDF)",
        "paste_cv": "O pega tu CV aquí",
        "paste_job": "📋 Pega la oferta (o sube un PDF arriba)",
        "paste_jobs": "📋 Pega las ofertas (separadas por ---)",
        "analyze": "Analizar",
        "match_result": "📊 Resultado del Match",
        "section_scores": "📐 Puntuación por Sección",
        "strengths_detected": "💪 Fortalezas Detectadas",
        "gaps_vs_job": "⚠️ Brechas vs Oferta",
        "gap_importance": "Importancia del Gap",
        "recommendations": "🧠 Recomendaciones",
        "no_cv_job": "Por favor proporciona un CV y una oferta de trabajo.",
        "none_detected": "Ninguna detectada",
        "highly_aligned": "¡Tu perfil está altamente alineado! Aplica con confianza 🎯",
        "consider_learning": "Considera aprender o reforzar: ",
        "improve_cv": "Reescribe tu CV para resaltar mejor la experiencia relevante.",
        "analysis_done": "Análisis completado 🚀",
        "weak_verbs": "✍️ Sugerencias de Verbos Débiles",
        "missing_sections": "📋 Secciones Faltantes en el CV",
        "keyword_density": "🔑 Densidad de Keywords",
        "cv_in_job": "CV", "job_in_job": "Oferta",
        "generic_cv_warning": "⚠️ Tu CV parece genérico — considera personalizarlo para esta oferta.",
        "experience_match": "🗓️ Experiencia",
        "language_match": "🌐 Idiomas",
        "education_match": "🎓 Formación",
        "cv_preview": "📄 Keywords Resaltadas en el CV",
        "history": "🕓 Historial de Análisis",
        "no_history": "No hay análisis previos.",
        "export_pdf": "📥 Exportar Informe como PDF",
        "multi_job_ranking": "📊 Ranking de Ofertas",
        "multi_cv_ranking": "📊 Ranking de CVs",
        "sector_detected": "🏷️ Sector Detectado",
        "years_cv": "Años en CV",
        "years_job": "Años requeridos",
        "languages_cv": "Idiomas en CV",
        "languages_job": "Idiomas en Oferta",
        "degree_cv": "Titulación en CV",
        "degree_job": "Titulación en Oferta",
    }
}

# ─────────────────────────────────────────────
# MODEL (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()

# ─────────────────────────────────────────────
# TEXT UTILITIES
# ─────────────────────────────────────────────
def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        return "\n".join(p.extract_text() or "" for p in reader.pages)
    except:
        return ""

def resolve_text(file, text_input=""):
    """
    Merge a PDF upload and a pasted text area into one string.
    - If only PDF  → use PDF text.
    - If only text → use text.
    - If both      → concatenate (PDF first) so neither source is lost.
    - If neither   → empty string (caller must validate).
    """
    pdf_text = extract_text_from_pdf(file).strip() if file else ""
    typed    = text_input.strip() if text_input else ""
    if pdf_text and typed:
        return pdf_text + "\n" + typed
    return pdf_text or typed

def has_cv(file, text_input=""):
    """True when at least one CV source has content."""
    return bool(resolve_text(file, text_input))

def has_job(file, text_input=""):
    """True when at least one job source has content."""
    return bool(resolve_text(file, text_input))

# FIX 2 — strip emails, URLs and LinkedIn handles BEFORE cleaning,
# so fragments like "emailcom" never enter the keyword pipeline.
def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"\S+@\S+\.\S+", " ", text)          # emails
    text = re.sub(r"https?://\S+", " ", text)           # URLs
    text = re.sub(r"linkedin\.com/\S+", " ", text)      # LinkedIn paths
    text = re.sub(r"[^a-z0-9áéíóúñü\s\+\#\.\-/]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def text_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

# ─────────────────────────────────────────────
# EMBEDDING WITH CACHE
# ─────────────────────────────────────────────
def get_embedding(text):
    h = text_hash(text)
    if "emb_cache" not in st.session_state:
        st.session_state["emb_cache"] = {}
    if h not in st.session_state["emb_cache"]:
        st.session_state["emb_cache"][h] = model.encode([text])[0]
    return st.session_state["emb_cache"][h]

def semantic_similarity(a, b):
    if not a or not b:
        return 0.0
    ea = get_embedding(a)
    eb = get_embedding(b)
    return float(cosine_similarity([ea], [eb])[0][0])

# ─────────────────────────────────────────────
# KEYWORD BANK
# ─────────────────────────────────────────────
def build_keyword_bank():
    return {
        "software_engineering": [
            "python","java","javascript","typescript","php","ruby","go","rust",
            "c","c++","c#","scala","kotlin","swift","node","nodejs","react","angular",
            "vue","svelte","nextjs","django","flask","spring","laravel","fastapi",
            "html","css","sass","tailwind","rest","graphql","microservices","oop",
            "sql","mysql","postgres","mongodb","redis","dynamodb","nosql",
            "aws","azure","gcp","docker","kubernetes","terraform","ansible","linux",
            "git","ci/cd","jenkins","github actions","tdd","jest","pytest",
        ],
        "data": [
            "data science","data analyst","analytics","machine learning","deep learning",
            "ai","artificial intelligence","python","r","sql","pandas","numpy",
            "tensorflow","pytorch","scikit learn","power bi","tableau","spark",
            "kafka","databricks","snowflake","bigquery","airflow",
        ],
        "cybersecurity": [
            "cybersecurity","pentesting","ethical hacking","siem","soc","firewall",
            "owasp","nmap","nessus","burp suite","kali","metasploit","forensics",
            "incident response","red team","blue team",
        ],
        "devops": [
            "devops","sre","docker","kubernetes","terraform","ansible","aws","azure",
            "gcp","jenkins","github actions","prometheus","grafana","elk",
            "microservices","scalability","high availability",
        ],
        "product": [
            "product manager","product owner","scrum","agile","kanban","backlog",
            "user stories","roadmap","sprint","stakeholders","ux research",
        ],
        "design": [
            "ux","ui","user experience","figma","sketch","adobe xd","photoshop",
            "illustrator","wireframes","prototyping","design system","usability",
        ],
        "marketing": [
            "seo","sem","google ads","facebook ads","google analytics","crm",
            "email marketing","copywriting","content marketing","branding","growth",
        ],
        "sales": [
            "sales","b2b","b2c","crm","salesforce","hubspot","negotiation",
            "pipeline","prospecting","lead generation",
        ],
        "hr": [
            "recruitment","talent acquisition","headhunting","onboarding","payroll",
            "compensation","hr","human resources",
        ],
        "finance": [
            "finance","accounting","budgeting","auditing","tax","compliance",
            "financial analysis","forecasting","treasury",
        ],
        "customer_service": [
            "customer support","helpdesk","zendesk","freshdesk","ticketing",
            "technical support","client support",
        ],
    }

SOFT_SKILLS = [
    "communication","teamwork","leadership","problem solving","critical thinking",
    "adaptability","creativity","time management","collaboration","presentation",
    "mentoring","coaching","decision making","analytical","detail oriented",
    "trabajo en equipo","liderazgo","resolución de problemas","comunicación",
    "adaptabilidad","creatividad","gestión del tiempo",
]

EQUIVALENCES = {
    "k8s": "kubernetes", "nodejs": "node", "node.js": "node",
    "reactjs": "react", "react.js": "react", "nextjs": "next.js",
    "scikitlearn": "scikit learn", "scikit-learn": "scikit learn",
    "postgresql": "postgres", "pgsql": "postgres",
}

def normalize_keyword(kw):
    return re.sub(r"[\.\-]", "", kw.lower().strip())

def unify_keywords(keywords):
    unified = set()
    for kw in keywords:
        n = normalize_keyword(kw)
        unified.add(EQUIVALENCES.get(n, n))
    return unified

def keyword_in_text(kw, text):
    """Return True if keyword (or a common inflection) appears in text."""
    pattern = r"\b" + re.escape(kw) + r"s?\b"
    return bool(re.search(pattern, text, flags=re.IGNORECASE))

def detect_sectors(text, top_n=2):
    """Return a list of the top_n most likely sectors for the given text."""
    bank = build_keyword_bank()
    text_lower = text.lower()
    scores = {}
    for sector, words in bank.items():
        hits = sum(1 for w in words if keyword_in_text(w, text_lower))
        scores[sector] = hits / max(len(words), 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [s[0] for s in ranked[:top_n]]

# ─────────────────────────────────────────────
# KEYWORD EXTRACTION
# ─────────────────────────────────────────────
def extract_items(text, sectors=None):
    """
    sectors: list of sector names to restrict the keyword bank.
    Falls back to all sectors when None or empty.
    """
    text_lower = text.lower()
    bank = build_keyword_bank()
    detected = set()

    sectors_to_use = sectors if sectors else list(bank.keys())
    for s in sectors_to_use:
        if s not in bank:
            continue
        for w in bank[s]:
            # FIX 3 applied here too — use inflection-tolerant check
            if keyword_in_text(w, text_lower):
                detected.add(w)

    for w in SOFT_SKILLS:
        if w in text_lower:
            detected.add(w)

    for c in re.findall(r"[a-z0-9\+\#\.\-/]{3,25}", text_lower):
        if any(x in c for x in ["js","ai","ml","net","ci/cd","k8s"]):
            detected.add(c)

    tech_list = sorted(unify_keywords(detected))
    return {"tech_text": " ".join(tech_list), "tech_list": tech_list}

# ─────────────────────────────────────────────
# LANGUAGE DETECTION
# ─────────────────────────────────────────────
LANGUAGE_KEYWORDS = {
    "english": ["english", "inglés", "ingles"],
    "spanish": ["spanish", "español", "castellano"],
    "french": ["french", "français", "frances", "francés"],
    "german": ["german", "deutsch", "alemán", "aleman"],
    "portuguese": ["portuguese", "portugués", "portugues"],
    "italian": ["italian", "italiano"],
    "chinese": ["chinese", "mandarin", "chino"],
    "arabic": ["arabic", "árabe", "arabe"],
    "dutch": ["dutch", "nederlands"],
    "russian": ["russian", "ruso"],
    "japanese": ["japanese", "japonés"],
}

def detect_languages(text):
    text_lower = text.lower()
    found = set()
    for lang, variants in LANGUAGE_KEYWORDS.items():
        if any(v in text_lower for v in variants):
            found.add(lang)
    return found

# ─────────────────────────────────────────────
# EXPERIENCE DETECTION
# ─────────────────────────────────────────────
def detect_years_of_experience(text):
    patterns = [
        r"(\d+)\+?\s*(?:years?|años?|yrs?)\s*(?:of\s*)?(?:experience|experiencia)",
        r"(\d+)\s*-\s*\d+\s*(?:years?|años?)\s*(?:of\s*)?(?:experience|experiencia)",
        r"(?:more than|más de|over)\s*(\d+)\s*(?:years?|años?)",
        r"(?:at least|minimum|mínimo|al menos)\s*(\d+)\s*(?:years?|años?)",
    ]
    years = []
    for p in patterns:
        for m in re.finditer(p, text.lower()):
            try:
                years.append(int(m.group(1)))
            except:
                pass
    return max(years) if years else None

# ─────────────────────────────────────────────
# EDUCATION DETECTION
# ─────────────────────────────────────────────
DEGREE_LEVELS = {
    "phd": 5, "doctorate": 5, "doctorado": 5,
    "master": 4, "máster": 4, "msc": 4, "mba": 4, "postgrado": 4, "postgraduate": 4,
    "bachelor": 3, "grado": 3, "licenciatura": 3, "degree": 3, "bsc": 3,
    "associate": 2, "fp superior": 2, "ciclo superior": 2,
    "fp": 1, "vocational": 1, "certificado": 1,
}

def detect_education(text):
    text_lower = text.lower()
    found = {}
    for deg, level in DEGREE_LEVELS.items():
        if deg in text_lower:
            found[deg] = level
    if not found:
        return None, 0
    best = max(found, key=lambda k: found[k])
    return best, found[best]

# ─────────────────────────────────────────────
# SEMANTIC CHUNKING
# ─────────────────────────────────────────────
def split_into_chunks(text, n_chunks=5):
    sentences = re.split(r"(?<=[.!\n])\s+", text)
    if len(sentences) < n_chunks:
        return [text]
    size = max(1, len(sentences) // n_chunks)
    chunks = []
    for i in range(0, len(sentences), size):
        chunk = " ".join(sentences[i:i + size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def chunked_similarity(cv_text, job_text):
    chunks = split_into_chunks(cv_text, n_chunks=5)
    if not chunks:
        return 0.0
    sims = [semantic_similarity(c, job_text) for c in chunks]
    return max(sims)

# ─────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────
def compute_section_scores(cv_text, job_text, cv_tech, job_tech,
                            cv_langs, job_langs, cv_deg, cv_deg_level,
                            job_deg, job_deg_level, cv_years, job_years, sector):
    # Technical
    cv_set = unify_keywords(cv_tech)
    job_set = unify_keywords(job_tech)
    tech_overlap = len(cv_set & job_set) / max(len(job_set), 1)
    tech_sem = chunked_similarity(" ".join(cv_tech), " ".join(job_tech))
    tech_score = round(((tech_overlap * 0.6) + (tech_sem * 0.4)) * 100, 1)

    # General / soft skills
    general_sim = chunked_similarity(cv_text, job_text)
    general_score = round(max(0, min(1, (general_sim - 0.3) / 0.5)) * 100, 1)

    # Languages
    lang_overlap = len(cv_langs & job_langs) / max(len(job_langs), 1) if job_langs else 1.0
    lang_score = round(lang_overlap * 100, 1)

    # Education
    if job_deg_level == 0:
        edu_score = 100.0
    elif cv_deg_level >= job_deg_level:
        edu_score = 100.0
    elif cv_deg_level == job_deg_level - 1:
        edu_score = 60.0
    else:
        edu_score = 20.0

    # Experience
    if job_years is None:
        exp_score = 100.0
    elif cv_years is None:
        exp_score = 50.0
    else:
        exp_score = round(min(cv_years / job_years, 1.0) * 100, 1)

    overall = round(
        tech_score * 0.40 +
        general_score * 0.20 +
        lang_score * 0.15 +
        edu_score * 0.10 +
        exp_score * 0.15,
        1
    )

    return {
        "overall": overall,
        "tech": tech_score,
        "general": general_score,
        "lang": lang_score,
        "edu": edu_score,
        "exp": exp_score,
    }

# ─────────────────────────────────────────────
# GAP ENGINE
# ─────────────────────────────────────────────
SEMANTIC_GROUPS = {
    "big data": {"spark", "hadoop", "bigquery", "airflow", "kafka", "databricks"},
    "nosql": {"mongodb", "cassandra", "redis", "dynamodb"},
    "data science": {"tensorflow", "pytorch", "scikit learn"},
    "cloud computing": {"aws", "azure", "gcp"},
    "testing": {"tdd", "jest", "pytest"},
}
GENERIC_TRASH_GAPS = {
    "engineering", "less", "excel", "office", "word", "powerpoint",
    "communication", "support", "education", "product design",
}

def logically_covered(kw, cv_set):
    for group, techs in SEMANTIC_GROUPS.items():
        if normalize_keyword(kw) == normalize_keyword(group):
            if any(t in cv_set for t in techs):
                return True
    return False

def compute_gaps_with_weight(cv_tech, job_tech, job_text):
    cv_set = unify_keywords(cv_tech)
    cv_text_joined = " ".join(cv_tech)
    job_set = unify_keywords(job_tech)
    job_lower = job_text.lower()
    gaps = []
    for kw in job_set:
        if kw in GENERIC_TRASH_GAPS:
            continue
        if kw in cv_set:
            continue
        # FIX 3 — also check plural/inflection before calling it a gap
        if keyword_in_text(kw, cv_text_joined):
            continue
        if logically_covered(kw, cv_set):
            continue
        freq = job_lower.count(kw)
        gaps.append((kw, freq))
    gaps.sort(key=lambda x: x[1], reverse=True)
    return gaps

# ─────────────────────────────────────────────
# CV QUALITY CHECKS
# ─────────────────────────────────────────────
WEAK_VERBS = {
    "participated in": "led / drove",
    "helped with": "delivered / contributed to",
    "was responsible for": "owned / managed",
    "worked on": "built / developed",
    "assisted": "supported / enabled",
    "involved in": "spearheaded / executed",
    "was part of": "contributed to / shaped",
    "made": "engineered / designed",
    # Spanish
    "participé en": "lideré / impulsé",
    "ayudé con": "entregué / contribuí a",
    "fui responsable de": "gestioné / dirigí",
    "trabajé en": "desarrollé / construí",
    "asistí": "apoyé / facilité",
    "estuve involucrado en": "lideré / ejecuté",
    "fui parte de": "contribuí a / formé parte clave de",
}

def detect_weak_verbs(text):
    text_lower = text.lower()
    found = {}
    for weak, strong in WEAK_VERBS.items():
        if weak in text_lower:
            found[weak] = strong
    return found

CV_SECTION_MARKERS = {
    "contact":    ["email", "phone", "linkedin", "contact", "contacto", "tel", "teléfono"],
    "experience": ["experience", "work", "employment", "experiencia", "trabajo", "trayectoria"],
    "education":  ["education", "studies", "university", "degree", "formación", "estudios", "universidad"],
    "skills":     ["skills", "technologies", "competencies", "habilidades", "competencias", "tecnologías"],
}

def detect_missing_sections(text):
    text_lower = text.lower()
    return [s for s, markers in CV_SECTION_MARKERS.items()
            if not any(m in text_lower for m in markers)]

def keyword_density(cv_text, job_tech):
    cv_lower = cv_text.lower()
    return {kw: cv_lower.count(kw) for kw in job_tech}

# ─────────────────────────────────────────────
# HIGHLIGHTED CV TEXT
# ─────────────────────────────────────────────
def highlight_cv_text(cv_text, cv_tech, job_tech):
    cv_set = set(cv_tech)
    job_set = set(job_tech)
    present = cv_set & job_set
    absent = job_set - cv_set

    highlighted = cv_text[:3000]
    for kw in sorted(present, key=len, reverse=True):
        highlighted = re.sub(
            re.escape(kw),
            f'<mark style="background:#b7f5b0;border-radius:3px;padding:1px 3px">{kw}</mark>',
            highlighted, flags=re.IGNORECASE
        )
    for kw in sorted(absent, key=len, reverse=True):
        highlighted = re.sub(
            re.escape(kw),
            f'<mark style="background:#ffd6d6;border-radius:3px;padding:1px 3px">{kw}</mark>',
            highlighted, flags=re.IGNORECASE
        )
    return highlighted.replace("\n", "<br>")

# ─────────────────────────────────────────────
# RADAR CHART
# ─────────────────────────────────────────────
def radar_chart(scores, lang):
    labels = ["Technical", "Soft Skills", "Languages", "Education", "Experience"]
    values = [scores["tech"], scores["general"], scores["lang"], scores["edu"], scores["exp"]]
    values_c = values + [values[0]]
    labels_c = labels + [labels[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_c, theta=labels_c, fill="toself",
        fillcolor="rgba(99,179,237,0.3)",
        line=dict(color="rgba(49,130,206,0.9)", width=2),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        margin=dict(l=40, r=40, t=40, b=40),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig

# ─────────────────────────────────────────────
# PDF EXPORT
# ─────────────────────────────────────────────

# Helvetica (built-in PDF font) only covers latin-1.
# Characters outside that range crash fpdf2 with "Not enough horizontal space".
# This helper replaces every out-of-range character with its closest ASCII
# equivalent or a safe fallback, keeping the text readable.
_ACCENT_MAP = str.maketrans({
    # lowercase
    'á':'a','à':'a','ä':'a','â':'a','ã':'a','å':'a',
    'é':'e','è':'e','ë':'e','ê':'e',
    'í':'i','ì':'i','ï':'i','î':'i',
    'ó':'o','ò':'o','ö':'o','ô':'o','õ':'o',
    'ú':'u','ù':'u','ü':'u','û':'u',
    'ý':'y','ÿ':'y','ñ':'n','ç':'c',
    # uppercase
    'Á':'A','À':'A','Ä':'A','Â':'A','Ã':'A','Å':'A',
    'É':'E','È':'E','Ë':'E','Ê':'E',
    'Í':'I','Ì':'I','Ï':'I','Î':'I',
    'Ó':'O','Ò':'O','Ö':'O','Ô':'O','Õ':'O',
    'Ú':'U','Ù':'U','Ü':'U','Û':'U',
    'Ý':'Y','Ñ':'N','Ç':'C',
})

def _pdf_safe(text: str) -> str:
    """Transliterate accented/special chars to ASCII-safe equivalents."""
    if not text:
        return ""
    text = text.translate(_ACCENT_MAP)
    # Drop anything still outside latin-1 (e.g. emojis, CJK)
    return text.encode("latin-1", errors="ignore").decode("latin-1")

def _pdf_cell(pdf, w, h, text, **kwargs):
    """pdf.cell wrapper that sanitizes text before rendering."""
    pdf.cell(w, h, _pdf_safe(text), **kwargs)

def _pdf_multi(pdf, w, h, text):
    """pdf.multi_cell wrapper that sanitizes text before rendering."""
    safe = _pdf_safe(text)
    if safe.strip():
        pdf.multi_cell(w, h, safe)

def generate_pdf_report(scores, gaps, strengths, recommendations, sector, lang_label):
    pdf = FPDF()
    pdf.set_margins(left=15, top=15, right=15)
    pdf.add_page()

    # Available content width (page width minus both margins)
    W = pdf.w - pdf.l_margin - pdf.r_margin   # ~180 mm for A4

    pdf.set_font("Helvetica", "B", 16)
    _pdf_cell(pdf, W, 10, "AI CV Analyzer - Match Report", ln=True, align="C")

    pdf.set_font("Helvetica", size=9)
    _pdf_cell(pdf, W, 7,
              f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Language: {lang_label}",
              ln=True, align="C")
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    _pdf_cell(pdf, W, 8, f"Overall Match: {scores['overall']}%", ln=True)
    _pdf_cell(pdf, W, 8, f"Sector: {_pdf_safe(sector)}", ln=True)
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 11)
    _pdf_cell(pdf, W, 8, "Scores by Section:", ln=True)
    pdf.set_font("Helvetica", size=10)
    for k, v in scores.items():
        if k != "overall":
            _pdf_cell(pdf, W, 7, f"  {k.capitalize()}: {v}%", ln=True)

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    _pdf_cell(pdf, W, 8, "Strengths:", ln=True)
    pdf.set_font("Helvetica", size=10)
    _pdf_multi(pdf, W, 7, ", ".join(strengths) if strengths else "None")

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    _pdf_cell(pdf, W, 8, "Gaps:", ln=True)
    pdf.set_font("Helvetica", size=10)
    _pdf_multi(pdf, W, 7, ", ".join(g[0] for g in gaps) if gaps else "None")

    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    _pdf_cell(pdf, W, 8, "Recommendations:", ln=True)
    pdf.set_font("Helvetica", size=10)
    for r in recommendations:
        _pdf_multi(pdf, W, 7, f"- {r}")

    return bytes(pdf.output())

# ─────────────────────────────────────────────
# FULL ANALYSIS PIPELINE
# ─────────────────────────────────────────────
def run_analysis(cv_text, job_text, lang):
    cv_clean = clean_text(cv_text)
    job_clean = clean_text(job_text)

    # FIX 1 — detect top-2 sectors from the job description
    sectors = detect_sectors(job_clean, top_n=2)
    primary_sector = sectors[0]   # used for display only

    cv_info = extract_items(cv_clean, sectors)
    job_info = extract_items(job_clean, sectors)

    cv_langs = detect_languages(cv_text)
    job_langs = detect_languages(job_text)
    cv_deg, cv_deg_level = detect_education(cv_text)
    job_deg, job_deg_level = detect_education(job_text)
    cv_years = detect_years_of_experience(cv_text)
    job_years = detect_years_of_experience(job_text)

    scores = compute_section_scores(
        cv_clean, job_clean,
        cv_info["tech_list"], job_info["tech_list"],
        cv_langs, job_langs,
        cv_deg, cv_deg_level,
        job_deg, job_deg_level,
        cv_years, job_years,
        primary_sector,
    )

    gaps = compute_gaps_with_weight(cv_info["tech_list"], job_info["tech_list"], job_text)
    weak_verbs = detect_weak_verbs(cv_text)
    missing_sections = detect_missing_sections(cv_text)
    # FIX 4 — density is now populated correctly because job_info uses multi-sector keywords
    density = keyword_density(cv_text.lower(), job_info["tech_list"])
    highlighted = highlight_cv_text(cv_text[:3000], cv_info["tech_list"], job_info["tech_list"])

    recs = []
    if gaps:
        recs.append(lang["consider_learning"] + ", ".join(g[0] for g in gaps[:5]))
    if scores["overall"] < 50:
        recs.append(lang["improve_cv"])
    common_kws = len(unify_keywords(cv_info["tech_list"]) & unify_keywords(job_info["tech_list"]))
    if scores["overall"] < 45 and common_kws < 3:
        recs.append(lang["generic_cv_warning"])
    if not recs:
        recs.append(lang["highly_aligned"])

    return {
        "sector": primary_sector,
        "sectors": sectors,
        "scores": scores,
        "strengths": cv_info["tech_list"],
        "job_tech": job_info["tech_list"],
        "gaps": gaps,
        "weak_verbs": weak_verbs,
        "missing_sections": missing_sections,
        "density": density,
        "highlighted": highlighted,
        "recommendations": recs,
        "cv_langs": cv_langs, "job_langs": job_langs,
        "cv_deg": cv_deg, "cv_deg_level": cv_deg_level,
        "job_deg": job_deg, "job_deg_level": job_deg_level,
        "cv_years": cv_years, "job_years": job_years,
    }

# ─────────────────────────────────────────────
# DISPLAY RESULTS
# ─────────────────────────────────────────────
def display_results(result, lang, lang_label, unique_key="default"):
    scores = result["scores"]
    gaps = result["gaps"]

    # Overview metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("🎯 Overall Match", f"{scores['overall']}%")
    # Show both detected sectors when there are two
    sectors_display = " + ".join(
        s.replace("_", " ").title() for s in result.get("sectors", [result["sector"]])
    )
    c2.metric(lang["sector_detected"], sectors_display)
    exp_str = f"{result['cv_years']}y" if result["cv_years"] else "—"
    job_exp_str = f"{result['job_years']}y req." if result["job_years"] else "—"
    c3.metric(lang["experience_match"], f"{exp_str} / {job_exp_str}")

    st.divider()

    # Radar + section scores
    col_radar, col_scores = st.columns([1.2, 1])
    with col_radar:
        fig = radar_chart(scores, lang)
        st.plotly_chart(fig, use_container_width=True)
    with col_scores:
        st.subheader(lang["section_scores"])
        section_labels = {
            "tech": "⚙️ Technical",
            "general": "💬 Soft Skills",
            "lang": "🌐 Languages",
            "edu": "🎓 Education",
            "exp": "🗓️ Experience",
        }
        for k, label in section_labels.items():
            val = scores[k]
            st.markdown(f"**{label}**: `{val}%`")
            st.progress(int(val) / 100)

    st.divider()

    # Profile details
    with st.expander("📋 Profile Details", expanded=False):
        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            st.markdown(f"**{lang['years_cv']}:** {result['cv_years'] or '—'}")
            st.markdown(f"**{lang['years_job']}:** {result['job_years'] or '—'}")
        with dc2:
            st.markdown(f"**{lang['languages_cv']}:** {', '.join(result['cv_langs']) or '—'}")
            st.markdown(f"**{lang['languages_job']}:** {', '.join(result['job_langs']) or '—'}")
        with dc3:
            st.markdown(f"**{lang['degree_cv']}:** {result['cv_deg'] or '—'}")
            st.markdown(f"**{lang['degree_job']}:** {result['job_deg'] or '—'}")

    # Strengths
    st.subheader(lang["strengths_detected"])
    if result["strengths"]:
        st.markdown(
            " ".join(
                f'<span style="background:#e6f4ea;border:1px solid #82c785;'
                f'border-radius:12px;padding:3px 10px;margin:2px;display:inline-block">{s}</span>'
                for s in result["strengths"]
            ),
            unsafe_allow_html=True,
        )
    else:
        st.info(lang["none_detected"])

    st.divider()

    # Gaps with importance bars
    st.subheader(lang["gaps_vs_job"])
    if gaps:
        max_freq = max((g[1] for g in gaps), default=1) or 1
        for kw, freq in gaps[:12]:
            col_g, col_b = st.columns([1, 3])
            col_g.markdown(f"`{kw}`")
            col_b.progress(freq / max_freq)
    else:
        st.success(lang["none_detected"])

    st.divider()

    # Keyword density
    with st.expander(lang["keyword_density"], expanded=False):
        density_data = sorted(
            [(kw, cnt) for kw, cnt in result["density"].items() if cnt > 0],
            key=lambda x: x[1], reverse=True
        )
        if density_data:
            df = pd.DataFrame(density_data[:20], columns=["Keyword", "Count in CV"])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info(lang["none_detected"])

    # Highlighted CV
    with st.expander(lang["cv_preview"], expanded=False):
        st.markdown(
            '<span style="background:#b7f5b0;padding:2px 8px;border-radius:4px">■ Present</span>&nbsp;'
            '<span style="background:#ffd6d6;padding:2px 8px;border-radius:4px">■ Missing</span>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size:13px;line-height:1.7;max-height:400px;overflow-y:auto;'
            f'border:1px solid #ddd;padding:12px;border-radius:8px">{result["highlighted"]}</div>',
            unsafe_allow_html=True,
        )

    # CV Quality
    st.divider()
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        with st.expander(lang["weak_verbs"], expanded=True):
            if result["weak_verbs"]:
                for weak, strong in result["weak_verbs"].items():
                    st.markdown(f'~~{weak}~~ → **{strong}**')
            else:
                st.success("✅ No weak verbs detected")
    with col_q2:
        with st.expander(lang["missing_sections"], expanded=True):
            if result["missing_sections"]:
                for s in result["missing_sections"]:
                    st.warning(f"Missing: **{s}**")
            else:
                st.success("✅ All key sections detected")

    # Recommendations
    st.divider()
    st.subheader(lang["recommendations"])
    for r in result["recommendations"]:
        st.write("• " + r)

    st.success(lang["analysis_done"])

    # PDF export
    pdf_bytes = generate_pdf_report(
        result["scores"], result["gaps"],
        result["strengths"], result["recommendations"],
        result["sector"], lang_label,
    )
    st.download_button(
        label=lang["export_pdf"],
        data=pdf_bytes,
        file_name="cv_match_report.pdf",
        mime="application/pdf",
        key=f"dl_pdf_{unique_key}",
    )

# ─────────────────────────────────────────────
# HISTORY
# ─────────────────────────────────────────────
def save_to_history(result, job_snippet):
    if "history" not in st.session_state:
        st.session_state["history"] = []
    entry = {
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "overall": result["scores"]["overall"],
        "sector": result["sector"],
        "job_snippet": job_snippet[:60] + "…",
        "gaps": [g[0] for g in result["gaps"][:5]],
    }
    st.session_state["history"].insert(0, entry)
    st.session_state["history"] = st.session_state["history"][:10]

def display_history(lang):
    st.subheader(lang["history"])
    if not st.session_state.get("history"):
        st.info(lang["no_history"])
        return
    for h in st.session_state["history"]:
        with st.expander(f"[{h['timestamp']}] {h['job_snippet']} — {h['overall']}%"):
            st.write(f"**Sector:** {h['sector']} | **Match:** {h['overall']}%")
            st.write(f"**Top gaps:** {', '.join(h['gaps']) or '—'}")

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
st.set_page_config(page_title="AI CV Matcher ", layout="wide", page_icon="🤖")

top_left, top_right = st.columns([2, 2])
with top_left:
    lang_choice = st.selectbox("🌍 Language / Idioma", ["English", "Español"])
with top_right:
    lang = LANGUAGES[lang_choice]
    mode = st.radio(lang["mode_select"], [lang["mode_candidate"], lang["mode_recruiter"]], horizontal=True)

st.title("🤖 AI CV Job Matcher")
st.divider()

# ── CANDIDATE MODE ──────────────────────────
if mode == lang["mode_candidate"]:
    tab_single, tab_multi_job, tab_history = st.tabs(
        ["Single Analysis", "Multi-Job Ranking", lang["history"]]
    )

    # ── Single analysis ──
    with tab_single:
        col_in, col_out = st.columns([1, 1.6])
        with col_in:
            st.markdown("**CV**")
            cv_file        = st.file_uploader(lang["upload_cv"], type=["pdf"], key="s_cv_file")
            cv_text_input  = st.text_area(lang["paste_cv"], height=180, key="s_cv_text")

            st.markdown("**Job description**")
            job_file       = st.file_uploader(lang["upload_job"], type=["pdf"], key="s_job_file")
            job_text_input = st.text_area(lang["paste_job"], height=180, key="s_job_text")

            analyze_btn = st.button(lang["analyze"], type="primary", use_container_width=True)

        with col_out:
            if analyze_btn:
                cv_text  = resolve_text(cv_file,  cv_text_input)
                job_text = resolve_text(job_file, job_text_input)
                if not cv_text or not job_text:
                    st.warning(lang["no_cv_job"])
                else:
                    with st.spinner("Analyzing…"):
                        result = run_analysis(cv_text, job_text, lang)
                    save_to_history(result, job_text)
                    display_results(result, lang, lang_choice, unique_key="single")

    # ── Multi-job ranking ──
    with tab_multi_job:
        st.markdown("Provide **one CV** and **multiple job descriptions** (paste separated by `---`, or upload PDFs).")
        col_mjl, col_mjr = st.columns(2)
        with col_mjl:
            st.markdown("**CV**")
            mj_cv_file  = st.file_uploader(lang["upload_cv"],  type=["pdf"], key="mj_cv_file")
            mj_cv_text  = st.text_area(lang["paste_cv"], height=150, key="mj_cv_text")
        with col_mjr:
            st.markdown("**Job descriptions (PDF uploads)**")
            mj_job_files = st.file_uploader(
                lang["upload_job"], type=["pdf"],
                accept_multiple_files=True, key="mj_job_files"
            )
            st.markdown("**Or paste (separate with `---`)**")
            mj_jobs_raw = st.text_area(lang["paste_jobs"], height=150, key="mj_jobs_text")

        mj_btn = st.button("🚀 " + lang["analyze"], key="mj_btn", use_container_width=True)

        if mj_btn:
            cv_text = resolve_text(mj_cv_file, mj_cv_text)
            # Build job list: uploaded PDFs first, then pasted blocks
            jobs = []
            if mj_job_files:
                for jf in mj_job_files:
                    t = extract_text_from_pdf(jf).strip()
                    if t:
                        jobs.append((jf.name, t))
            if mj_jobs_raw.strip():
                for i, block in enumerate(mj_jobs_raw.split("---"), 1):
                    block = block.strip()
                    if block:
                        jobs.append((f"Pasted job {i}", block))

            if not cv_text or not jobs:
                st.warning(lang["no_cv_job"])
            else:
                rows = []
                for label, job_text in jobs:
                    with st.spinner(f"Analyzing: {label}…"):
                        r = run_analysis(cv_text, job_text, lang)
                    save_to_history(r, job_text)
                    rows.append({
                        "Job": label,
                        "Snippet": job_text[:55] + "…",
                        "Overall %": r["scores"]["overall"],
                        "Tech %": r["scores"]["tech"],
                        "Sector": " + ".join(r.get("sectors", [r["sector"]])),
                        "Top gaps": ", ".join(g[0] for g in r["gaps"][:3]),
                    })
                df = pd.DataFrame(rows).sort_values("Overall %", ascending=False).reset_index(drop=True)
                df.insert(0, "Rank", range(1, len(df) + 1))
                st.subheader(lang["multi_job_ranking"])
                st.dataframe(df, use_container_width=True, hide_index=True)

    with tab_history:
        display_history(lang)

# ── RECRUITER MODE ──────────────────────────
else:
    st.info("🏢 **Recruiter mode** — match multiple CVs against one job description.")
    tab_multi_cv, tab_hist_r = st.tabs(["Multi-CV Ranking", lang["history"]])

    with tab_multi_cv:
        st.markdown("**Job description**")
        r_job_file  = st.file_uploader(lang["upload_job"], type=["pdf"], key="r_job_file")
        r_job_text  = st.text_area(lang["paste_job"], height=160, key="r_job_text")

        st.markdown("**CVs — upload PDFs and/or paste text blocks separated by `---`**")
        col_rl, col_rr = st.columns(2)
        with col_rl:
            cv_files = st.file_uploader(
                lang["upload_cvs"], type=["pdf"],
                accept_multiple_files=True, key="r_cvs"
            )
        with col_rr:
            r_cv_pasted = st.text_area("Paste CV text blocks (separate with ---)", height=160, key="r_cv_text")

        r_btn = st.button("🚀 " + lang["analyze"], key="r_btn", use_container_width=True)

        if r_btn:
            job_text = resolve_text(r_job_file, r_job_text)

            # Build CV list: uploaded PDFs + pasted blocks
            cvs = []
            if cv_files:
                for cf in cv_files:
                    t = extract_text_from_pdf(cf).strip()
                    if t:
                        cvs.append((cf.name, t))
            if r_cv_pasted.strip():
                for i, block in enumerate(r_cv_pasted.split("---"), 1):
                    block = block.strip()
                    if block:
                        cvs.append((f"Pasted CV {i}", block))

            if not job_text or not cvs:
                st.warning(lang["no_cv_job"])
            else:
                rows = []
                results_map = {}
                for label, cv_text in cvs:
                    with st.spinner(f"Analyzing {label} ({len(rows)+1}/{len(cvs)})…"):
                        r = run_analysis(cv_text, job_text, lang)
                    save_to_history(r, job_text)
                    results_map[label] = r
                    rows.append({
                        "CV": label,
                        "Overall %": r["scores"]["overall"],
                        "Tech %": r["scores"]["tech"],
                        "Lang %": r["scores"]["lang"],
                        "Edu %": r["scores"]["edu"],
                        "Exp %": r["scores"]["exp"],
                        "Sector": " + ".join(r.get("sectors", [r["sector"]])),
                        "Top gaps": ", ".join(g[0] for g in r["gaps"][:3]),
                    })
                df = pd.DataFrame(rows).sort_values("Overall %", ascending=False).reset_index(drop=True)
                df.insert(0, "Rank", range(1, len(df) + 1))
                st.subheader(lang["multi_cv_ranking"])
                st.dataframe(df, use_container_width=True, hide_index=True)

                st.divider()
                st.markdown("### Detailed reports")
                for row in df.to_dict("records"):
                    with st.expander(f"#{row['Rank']} {row['CV']} — {row['Overall %']}%"):
                        display_results(results_map[row["CV"]], lang, lang_choice,
                                        unique_key=f"rec_{row['Rank']}_{row['CV']}")

    with tab_hist_r:
        display_history(lang)
