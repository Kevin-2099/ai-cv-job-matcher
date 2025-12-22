import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
import re

# ------------------------------
# LANGUAGE SELECTOR
# ------------------------------
LANGUAGES = {
    "English": {
        "upload_cv": "📄 Upload CV (PDF optional)",
        "paste_cv": "Or paste CV text",
        "paste_job": "📋 Paste job description",
        "analyze": "Analyze",
        "match_result": "📊 Match Result",
        "strengths_detected": "💪 Strengths Detected",
        "gaps_vs_job": "⚠️ Gaps vs Job Offer",
        "recommendations": "🧠 Recommendations",
        "no_cv_job": "Please provide a CV and a job description to analyze.",
        "none_detected": "None detected",
        "highly_aligned": "Your profile is highly aligned! Apply with confidence 🎯",
        "consider_learning": "Consider learning or reinforcing: ",
        "improve_cv": "Improve your CV to better highlight relevant experience.",
        "analysis_done": "Analysis completed 🚀"
    },
    "Español": {
        "upload_cv": "📄 Subir CV (PDF opcional)",
        "paste_cv": "O pega tu CV aquí",
        "paste_job": "📋 Pega la oferta de trabajo",
        "analyze": "Analizar",
        "match_result": "📊 Resultado del Match",
        "strengths_detected": "💪 Fortalezas Detectadas",
        "gaps_vs_job": "⚠️ Brechas vs Oferta",
        "recommendations": "🧠 Recomendaciones",
        "no_cv_job": "Por favor proporciona un CV y una descripción de la oferta para analizar.",
        "none_detected": "Ninguna detectada",
        "highly_aligned": "¡Tu perfil está altamente alineado! Aplica con confianza 🎯",
        "consider_learning": "Considera aprender o reforzar: ",
        "improve_cv": "Mejora tu CV para resaltar experiencia relevante.",
        "analysis_done": "Análisis completado 🚀"
    }
}

# ------------------------------
# LOAD MODEL
# ------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = load_model()

# ------------------------------
# HELPERS
# ------------------------------
def extract_text_from_pdf(file):
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
        return text
    except:
        return ""

def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9áéíóúñü\s\+\#\.\-/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ------------------------------
# MEGA DICTIONARY
# ------------------------------
def build_keyword_bank():
    return {
        "software_engineering": [
            "python","java","javascript","typescript","php","ruby","go","rust",
            "c","c++","c#","scala","perl","objective-c","kotlin","swift",
            "node","nodejs","node.js","react","reactjs","react.js",
            "angular","vue","svelte","nextjs","next.js",
            "django","flask","spring","spring boot","laravel","symfony","express","fastapi","nest",
            "html","css","sass","less","bootstrap","tailwind",
            "oop","functional programming","microservices","rest","graphql","soap",
            "sql","mysql","postgres","postgresql","oracle","mariadb","sqlite",
            "nosql","mongodb","cassandra","redis","dynamodb",
            "aws","azure","gcp","google cloud","cloud computing","serverless","lambda",
            "docker","kubernetes","k8s","terraform","ansible","pulumi","helm",
            "linux","bash","shell scripting",
            "git","github","gitlab","bitbucket","ci/cd","jenkins","github actions",
            "testing","unit testing","integration testing","tdd","bdd","jest","pytest"
        ],
        "data": [
            "data science","data scientist","data analyst","analytics",
            "big data","data engineering","data engineer",
            "machine learning","deep learning","ai","artificial intelligence",
            "python","r","sql","pandas","numpy",
            "tensorflow","pytorch","keras","huggingface","scikit learn","scikitlearn",
            "power bi","tableau","qlik",
            "hadoop","spark","pyspark","kafka","databricks","snowflake","bigquery","airflow","hive"
        ],
        "cybersecurity": [
            "cybersecurity","security","cyber security",
            "pentesting","penetration testing","ethical hacking",
            "siem","soc","firewall","waf",
            "owasp","nmap","nessus","burp suite","kali","metasploit",
            "forensics","incident response","blue team","red team"
        ],
        "devops": [
            "devops","sre","site reliability engineering",
            "docker","kubernetes","terraform","ansible","chef","puppet",
            "aws","azure","gcp",
            "jenkins","github actions","gitlab ci",
            "prometheus","grafana","elastic","logstash","kibana",
            "microservices","scalability","high availability"
        ],
        "product": [
            "product manager","product owner","product management",
            "scrum","agile","kanban",
            "backlog","user stories","roadmap","sprint",
            "stakeholders","ux research","product discovery"
        ],
        "design": [
            "ux","ui","ux/ui","user experience","user interface",
            "product design","graphic design",
            "figma","sketch","adobe xd","photoshop","illustrator",
            "wireframes","prototyping","design system","usability testing"
        ],
        "marketing": [
            "marketing","digital marketing","growth marketing",
            "seo","sem","google ads","facebook ads","tiktok ads",
            "analytics","google analytics","tag manager",
            "crm","email marketing","copywriting","content marketing","branding"
        ],
        "sales": [
            "sales","ventas","inside sales","outside sales",
            "b2b","b2c","consultative selling",
            "crm","salesforce","hubspot",
            "negotiation","pipeline","prospecting","closing","lead generation"
        ],
        "hr": [
            "hr","human resources","recursos humanos",
            "recruitment","reclutamiento","talent acquisition",
            "headhunting","onboarding","payroll","compensation & benefits"
        ],
        "finance": [
            "finance","finanzas","accounting","contabilidad",
            "budgeting","auditing","tax","compliance",
            "excel","financial analysis","forecasting","treasury"
        ],
        "administration": [
            "office","office manager","administration","administrativo",
            "assistant","administrative assistant",
            "excel","word","powerpoint","ms office",
            "documentation","invoices","support"
        ],
        "customer_service": [
            "customer support","customer service","soporte",
            "helpdesk","technical support",
            "ticketing","zendesk","freshdesk",
            "communication","client support"
        ],
        "education": [
            "teacher","profesor","education","educación",
            "training","learning","teaching",
            "curriculum","pedagogy"
        ],
        "healthcare": [
            "healthcare","sanidad","medical","medicina",
            "doctor","nurse","clinician",
            "patients","clinical records","emergency"
        ]
    }

# ------------------------------
# NORMALIZATION & EQUIVALENCES
# ------------------------------
def normalize_keyword(kw):
    kw = kw.lower().strip()
    kw = kw.replace(".", "").replace("-", "")
    return kw

EQUIVALENCES = {
    "k8s": "kubernetes",
    "nodejs": "node",
    "nodejs.": "node",
    "node.js": "node",
    "reactjs": "react",
    "react.js": "react",
    "nextjs": "next.js",
    "nextjs.": "next.js",
    "next.js.": "next.js",
}

def unify_keywords(keywords):
    unified = set()
    for kw in keywords:
        kw_norm = normalize_keyword(kw)
        if kw_norm in EQUIVALENCES:
            unified.add(EQUIVALENCES[kw_norm])
        else:
            unified.add(kw_norm)
    return unified

# ------------------------------
# EXTRACTOR
# ------------------------------
def extract_items(text):
    text = text.lower()
    keywords = build_keyword_bank()
    detected_keywords = set()

    for _, words in keywords.items():
        for w in words:
            if w in text:
                detected_keywords.add(w)

    candidates = re.findall(r"[a-z0-9\+\#\.\-/]{3,25}", text)
    for c in candidates:
        if any(x in c for x in ["js","ai","ml","net","ci/cd","k8s"]) and len(c) > 2:
            detected_keywords.add(c)

    tech_list = sorted(unify_keywords(detected_keywords))
    return {
        "tech_text": " ".join(tech_list),
        "tech_list": tech_list
    }

# ------------------------------
# SIMILARITY + SCORING
# ------------------------------
def semantic_similarity(a, b):
    if not a or not b:
        return 0
    emb1 = model.encode([a])
    emb2 = model.encode([b])
    return float(cosine_similarity(emb1, emb2)[0][0])

def combined_match_score(cv_text, job_text, cv_tech_text, job_tech_text):
    general_sim = semantic_similarity(cv_text, job_text)
    tech_sim = semantic_similarity(cv_tech_text, job_tech_text)
    final_score = (general_sim * 0.4) + (tech_sim * 0.6)
    normalized = max(0, min(1, (final_score - 0.4) / 0.5))
    return round(normalized * 100, 2)

# ------------------------------
# SMART GAP ENGINE
# ------------------------------
SEMANTIC_GROUPS = {
    "big data": {"spark", "hadoop", "bigquery", "airflow", "kafka", "databricks", "hive"},
    "nosql": {"mongodb", "cassandra", "redis", "dynamodb"},
    "data science": {"tensorflow", "pytorch", "keras", "huggingface", "scikit learn", "scikitlearn"},
    "serverless": {"lambda"},
    "cloud computing": {"aws", "azure", "gcp"},
    "testing": {"unit testing", "tdd", "jest", "pytest"}
}

GENERIC_TRASH_GAPS = {
    "engineering","ingenieria",
    "less","excel","office","word","powerpoint","ms office",
    "communication","support","helpdesk","customer service",
    "education","healthcare","product design"
}

def logically_covered(required_word, cv_set):
    required_word_norm = normalize_keyword(required_word)
    for group, techs in SEMANTIC_GROUPS.items():
        if required_word_norm == normalize_keyword(group):
            if any(t in cv_set for t in techs):
                return True
    return False

def compute_gaps(cv_tech, job_tech):
    cv_set = set(unify_keywords(cv_tech))
    job_set = set(unify_keywords(job_tech))
    gaps = []
    for kw in job_set:
        if kw in GENERIC_TRASH_GAPS:
            continue
        if kw in cv_set:
            continue
        if logically_covered(kw, cv_set):
            continue
        gaps.append(kw)
    return gaps

# ------------------------------
# UI
# ------------------------------
st.set_page_config(page_title="AI CV Matcher", layout="wide")  # layout wide para más espacio horizontal
lang_choice = st.selectbox("Select language / Selecciona idioma", ["English", "Español"])
lang = LANGUAGES[lang_choice]

st.title("🤖 AI CV Analyzer + Job Matching")

# Creamos dos columnas: izquierda para inputs, derecha para resultados
col_input, col_output = st.columns([1, 1.5])  # Ajusta proporción si quieres más espacio a la derecha

with col_input:
    cv_file = st.file_uploader(lang["upload_cv"], type=["pdf"])
    cv_text_input = st.text_area(lang["paste_cv"], height=300)
    job_text = st.text_area(lang["paste_job"], height=350)
    analyze_pressed = st.button(lang["analyze"])

with col_output:
    # Solo mostramos resultados si se presionó el botón
    if analyze_pressed:
        if not job_text or (not cv_file and not cv_text_input):
            st.warning(lang["no_cv_job"])
        else:
            cv_text = extract_text_from_pdf(cv_file) if cv_file else cv_text_input
            cv_text_clean = clean_text(cv_text)
            job_text_clean = clean_text(job_text)

            cv_info = extract_items(cv_text_clean)
            job_info = extract_items(job_text_clean)

            score = combined_match_score(
                cv_text_clean,
                job_text_clean,
                cv_info["tech_text"],
                job_info["tech_text"]
            )

            st.subheader(lang["match_result"])
            st.metric("Match", f"{score}%")

            st.subheader(lang["strengths_detected"])
            st.write(", ".join(cv_info["tech_list"]) or lang["none_detected"])

            st.subheader(lang["gaps_vs_job"])
            gaps_list = compute_gaps(cv_info["tech_list"], job_info["tech_list"])
            st.write(", ".join(gaps_list) if gaps_list else lang["none_detected"])

            st.subheader(lang["recommendations"])
            recs = []
            if gaps_list:
                recs.append(lang["consider_learning"] + ", ".join(gaps_list))
            if score < 60:
                recs.append(lang["improve_cv"])
            if not recs:
                recs.append(lang["highly_aligned"])

            for r in recs:
                st.write("• " + r)

            st.success(lang["analysis_done"])
