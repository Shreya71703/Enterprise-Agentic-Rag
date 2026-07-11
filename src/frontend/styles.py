"""
Stylesheets for Enterprise RAG Streamlit App

Separates visual style CSS tokens from app logic to prevent python tokenization issues
and ensure clean modular code.
"""

css_dark = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Premium cool glassmorphism base */
.stApp {
    background: radial-gradient(circle at 50% 50%, #0d1527 0%, #070a12 100%);
    color: #f8fafc;
    font-family: 'Outfit', sans-serif;
}
.stSidebar {
    background-color: #080c15;
    border-right: 1px solid #1a2436;
}

/* Global text visibility override to make all plain texts bright and readable */
.stApp p, .stApp li, .stApp label, .stApp span {
    color: #e2e8f0 !important; /* bright grey-white */
    font-size: 1rem;
}

h1, h2, h3, h4, h5, h6 {
    color: #c084fc !important; /* bright purple/violet */
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    letter-spacing: -0.02em;
    text-shadow: 0 0 12px rgba(192, 132, 252, 0.2);
}

/* Brightened Sidebar elements */
.stSidebar p, .stSidebar span, .stSidebar label, .stSidebar div {
    color: #e2e8f0 !important;
    font-size: 0.95rem;
}
.stSidebar code {
    color: #38bdf8 !important;
    background-color: #0f172a !important;
}
.stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4 {
    color: #c084fc !important;
}

/* Glowing metric card containers with high contrast labels */
div[data-testid="metric-container"] {
    background: rgba(15, 23, 42, 0.7);
    backdrop-filter: blur(12px) saturate(180%);
    border: 1px solid rgba(129, 140, 248, 0.25);
    border-radius: 12px;
    padding: 1.25rem 1.5rem !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4);
    transition: all 0.3s ease;
}
div[data-testid="metric-container"]:hover {
    border-color: rgba(129, 140, 248, 0.6);
    box-shadow: 0 8px 32px 0 rgba(129, 140, 248, 0.3);
    transform: translateY(-2px);
}
div[data-testid="stMetricValue"] {
    color: #38bdf8 !important; /* bright cyan */
    font-family: 'Outfit', sans-serif;
    font-size: 2rem !important;
    font-weight: 700;
    text-shadow: 0 0 8px rgba(56, 189, 248, 0.2);
}
div[data-testid="stMetricLabel"] > div {
    color: #cbd5e1 !important; /* bright cool grey */
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Styled Chat Interface */
div[data-testid="stChatMessage"] {
    background: rgba(15, 23, 42, 0.8) !important;
    border: 1px solid rgba(148, 163, 184, 0.2) !important;
    border-radius: 14px !important;
    margin-bottom: 12px !important;
    padding: 1.25rem !important;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.25);
}
div[data-testid="stChatMessage"][data-testid="stChatMessageAssistant"] {
    border-left: 4px solid #c084fc !important;
}
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] span,
div[data-testid="stChatMessage"] li {
    color: #f1f5f9 !important;
    font-size: 1rem;
    line-height: 1.6;
}

/* Input field customization */
div[data-testid="stChatInput"] textarea {
    color: #f8fafc !important;
    background-color: #0f172a !important;
    border: 1px solid rgba(148, 163, 184, 0.35) !important;
    border-radius: 8px !important;
}

/* Buttons customization */
.stButton>button {
    background: linear-gradient(135deg, #a855f7 0%, #6366f1 100%);
    color: #ffffff !important;
    border-radius: 8px;
    border: none;
    padding: 0.55rem 1.6rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #c084fc 0%, #818cf8 100%);
    box-shadow: 0 0 20px rgba(168, 85, 247, 0.75);
    transform: translateY(-1px);
}

/* Expander overrides */
details, 
div[data-testid="stExpander"], 
div[data-testid="stExpander"] details {
    background-color: #0f172a !important;
    background: #0f172a !important;
    border: 1px solid rgba(148, 163, 184, 0.25) !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3) !important;
}
summary,
div[data-testid="stExpander"] summary {
    background-color: #0f172a !important;
    background: #0f172a !important;
    color: #c084fc !important; /* bright purple header text */
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    border-radius: 8px !important;
}
summary:hover {
    background-color: #1e293b !important;
}
details > div,
div[data-testid="stExpander"] details > div {
    background-color: #0f172a !important;
    background: #0f172a !important;
    padding: 1rem !important;
    color: #e2e8f0 !important;
}
div[data-testid="stExpander"] p,
div[data-testid="stExpander"] span,
div[data-testid="stExpander"] li,
div[data-testid="stExpander"] code,
div[data-testid="stExpander"] div {
    color: #e2e8f0 !important;
}

/* Tabs selector overrides */
div[data-testid="stTabs"] button p {
    color: #cbd5e1 !important; /* bright silver text for unselected tabs */
    font-size: 1.05rem !important;
    font-weight: 600 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] p {
    color: #c084fc !important; /* bright purple text for selected tab */
    font-weight: 700 !important;
}

/* Code and Terminal styling */
code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
    background-color: #0b0f19 !important;
    padding: 0.15rem 0.4rem !important;
    border-radius: 4px !important;
    color: #f472b6 !important;
}
.stCodeBlock pre {
    background-color: #070a13 !important;
    border: 1px solid rgba(148, 163, 184, 0.15) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}

/* Execution trace badge colors */
.trace-badge-call {
    background-color: rgba(168, 85, 247, 0.2);
    color: #c084fc;
    border: 1px solid rgba(168, 85, 247, 0.4);
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
}
.trace-badge-result {
    background-color: rgba(56, 189, 248, 0.15);
    color: #38bdf8;
    border: 1px solid rgba(56, 189, 248, 0.35);
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
}
</style>
"""

css_light = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Premium Light Mode Base */
.stApp {
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
    color: #0f172a;
    font-family: 'Outfit', sans-serif;
}
.stSidebar {
    background-color: #f1f5f9;
    border-right: 1px solid #e2e8f0;
}

/* Global text visibility override to make all plain texts crisp and readable */
.stApp p, .stApp li, .stApp label, .stApp span {
    color: #334155 !important; /* crisp slate charcoal */
    font-size: 1rem;
}

h1, h2, h3, h4, h5, h6 {
    color: #4f46e5 !important; /* elegant royal indigo */
    font-family: 'Outfit', sans-serif;
    font-weight: 700;
    letter-spacing: -0.02em;
}

/* Sidebar elements in Light Mode */
.stSidebar p, .stSidebar span, .stSidebar label, .stSidebar div {
    color: #334155 !important;
    font-size: 0.95rem;
}
.stSidebar code {
    color: #0f172a !important;
    background-color: #e2e8f0 !important;
}
.stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4 {
    color: #4f46e5 !important;
}

/* Glowing metric card containers with high contrast labels */
div[data-testid="metric-container"] {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(12px) saturate(180%);
    border: 1px solid rgba(79, 70, 229, 0.15);
    border-radius: 12px;
    padding: 1.25rem 1.5rem !important;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease;
}
div[data-testid="metric-container"]:hover {
    border-color: rgba(79, 70, 229, 0.4);
    box-shadow: 0 8px 32px 0 rgba(79, 70, 229, 0.12);
    transform: translateY(-2px);
}
div[data-testid="stMetricValue"] {
    color: #2563eb !important; /* deep sapphire blue */
    font-family: 'Outfit', sans-serif;
    font-size: 2rem !important;
    font-weight: 700;
}
div[data-testid="stMetricLabel"] > div {
    color: #4f46e5 !important; /* royal indigo */
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Styled Chat Interface */
div[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.75) !important;
    border: 1px solid rgba(79, 70, 229, 0.1) !important;
    border-radius: 14px !important;
    margin-bottom: 12px !important;
    padding: 1.25rem !important;
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04);
}
div[data-testid="stChatMessage"][data-testid="stChatMessageAssistant"] {
    border-left: 4px solid #4f46e5 !important;
    background: rgba(79, 70, 229, 0.03) !important;
}
div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] span,
div[data-testid="stChatMessage"] li {
    color: #1e293b !important;
    font-size: 1rem;
    line-height: 1.6;
}

/* Input field customization */
div[data-testid="stChatInput"] textarea {
    color: #0f172a !important;
    background-color: #ffffff !important;
    border: 1px solid rgba(79, 70, 229, 0.25) !important;
    border-radius: 8px !important;
}

/* Buttons customization */
.stButton>button {
    background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
    color: #ffffff !important;
    border-radius: 8px;
    border: none;
    padding: 0.55rem 1.6rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    background: linear-gradient(135deg, #6366f1 0%, #2563eb 100%);
    box-shadow: 0 0 16px rgba(79, 70, 229, 0.4);
    transform: translateY(-1px);
}

/* Expander overrides */
details, 
div[data-testid="stExpander"], 
div[data-testid="stExpander"] details {
    background-color: #ffffff !important;
    background: #ffffff !important;
    border: 1px solid rgba(79, 70, 229, 0.15) !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.04) !important;
}
summary,
div[data-testid="stExpander"] summary {
    background-color: #ffffff !important;
    background: #ffffff !important;
    color: #4f46e5 !important; /* royal indigo header text */
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    border-radius: 8px !important;
}
summary:hover {
    background-color: #f8fafc !important;
}
details > div,
div[data-testid="stExpander"] details > div {
    background-color: #ffffff !important;
    background: #ffffff !important;
    padding: 1rem !important;
    color: #334155 !important;
}
div[data-testid="stExpander"] p,
div[data-testid="stExpander"] span,
div[data-testid="stExpander"] li,
div[data-testid="stExpander"] code,
div[data-testid="stExpander"] div {
    color: #334155 !important;
}

/* Tabs selector overrides */
div[data-testid="stTabs"] button p {
    color: #475569 !important; /* unselected tab */
    font-size: 1.05rem !important;
    font-weight: 600 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] p {
    color: #4f46e5 !important; /* selected tab */
    font-weight: 700 !important;
}

/* Code and Terminal styling */
code {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.9rem !important;
    background-color: #f1f5f9 !important;
    padding: 0.15rem 0.4rem !important;
    border-radius: 4px !important;
    color: #be185d !important; /* dark pink */
}
.stCodeBlock pre {
    background-color: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 1rem !important;
}

/* Execution trace badge colors */
.trace-badge-call {
    background-color: rgba(79, 70, 229, 0.1);
    color: #4f46e5;
    border: 1px solid rgba(79, 70, 229, 0.35);
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
}
.trace-badge-result {
    background-color: rgba(37, 99, 235, 0.08);
    color: #2563eb;
    border: 1px solid rgba(37, 99, 235, 0.25);
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    font-size: 0.8rem;
    font-family: 'JetBrains Mono', monospace;
}
</style>
"""
