from typing import Optional

import streamlit as st
import streamlit.components.v1 as components

THEME_OPTIONS = ["Light Mode", "Dark Mode"]
_LEGACY_THEME_MAP = {
    "light": "Light Mode",
    "light mode": "Light Mode",
    "night": "Dark Mode",
    "dark": "Dark Mode",
    "dark mode": "Dark Mode",
    "navy": "Dark Mode",
    "navy blue": "Dark Mode",
}


def normalize_theme(value: Optional[str]) -> str:
    """Map legacy theme names and free-form input to a supported option."""
    if not value:
        return "Dark Mode"
    if value in THEME_OPTIONS:
        return value
    return _LEGACY_THEME_MAP.get(value.strip().lower(), "Dark Mode")


_THEME_PALETTES = {
    "Light Mode": {
        "background": "#f4f6fb",
        "surface": "#ffffff",
        "surface_alt": "#eef3fb",
        "text": "#121a2b",
        "muted": "#5b6474",
        "border": "#d7dfed",
        "accent": "#2563eb",
        "accent_soft": "#93c5fd",
        "accent_text": "#ffffff",
        "highlight": "#7c3aed",
        "shadow": "rgba(15, 23, 42, 0.10)",
        "glow": "rgba(37, 99, 235, 0.18)",
        "bg_gradient": "radial-gradient(1200px 600px at 85% -10%, rgba(124,58,237,0.10), transparent 60%), radial-gradient(900px 500px at -10% 20%, rgba(37,99,235,0.12), transparent 55%), #f4f6fb",
    },
    "Dark Mode": {
        "background": "#0b1020",
        "surface": "#121a2e",
        "surface_alt": "#182342",
        "text": "#e8edfb",
        "muted": "#9aa6c2",
        "border": "#233158",
        "accent": "#60a5fa",
        "accent_soft": "#3b82f6",
        "accent_text": "#08111f",
        "highlight": "#a78bfa",
        "shadow": "rgba(2, 6, 23, 0.55)",
        "glow": "rgba(96, 165, 250, 0.25)",
        "bg_gradient": "radial-gradient(1100px 600px at 90% -5%, rgba(167,139,250,0.18), transparent 60%), radial-gradient(900px 500px at -10% 10%, rgba(96,165,250,0.18), transparent 55%), #0b1020",
    },
}


def _get_palette(theme_mode: str) -> dict:
    return _THEME_PALETTES.get(normalize_theme(theme_mode), _THEME_PALETTES["Dark Mode"])


def _inject_cursor_glow(colors: dict) -> None:
    """Attach a soft cursor-following glow to the parent Streamlit document."""
    glow_color = colors["glow"]
    accent = colors["accent"]
    components.html(
        f"""
        <script>
        (function() {{
          try {{
            const doc = window.parent.document;
            if (!doc) return;
            const existing = doc.getElementById('cursor-glow');
            if (existing) existing.remove();

            const glow = doc.createElement('div');
            glow.id = 'cursor-glow';
            glow.style.cssText = [
              'position:fixed',
              'top:0',
              'left:0',
              'width:320px',
              'height:320px',
              'border-radius:50%',
              'pointer-events:none',
              'z-index:1',
              'transform:translate(-50%,-50%)',
              'background:radial-gradient(circle, {glow_color} 0%, rgba(0,0,0,0) 65%)',
              'mix-blend-mode:screen',
              'filter:blur(10px)',
              'opacity:0',
              'transition:opacity 300ms ease-out, transform 120ms ease-out'
            ].join(';');
            doc.body.appendChild(glow);

            const dot = doc.createElement('div');
            dot.id = 'cursor-dot';
            dot.style.cssText = [
              'position:fixed',
              'top:0',
              'left:0',
              'width:6px',
              'height:6px',
              'border-radius:50%',
              'pointer-events:none',
              'z-index:2',
              'transform:translate(-50%,-50%)',
              'background:{accent}',
              'box-shadow:0 0 8px {accent}',
              'opacity:0',
              'transition:opacity 200ms ease-out, transform 80ms ease-out'
            ].join(';');
            doc.body.appendChild(dot);

            let mouseX = 0, mouseY = 0;
            let glowX = 0, glowY = 0;
            function onMove(e) {{
              mouseX = e.clientX;
              mouseY = e.clientY;
              glow.style.opacity = 0.5;
              dot.style.opacity = 0.55;
              dot.style.left = mouseX + 'px';
              dot.style.top = mouseY + 'px';
            }}
            function onLeave() {{
              glow.style.opacity = 0;
              dot.style.opacity = 0;
            }}
            function tick() {{
              glowX += (mouseX - glowX) * 0.12;
              glowY += (mouseY - glowY) * 0.12;
              glow.style.left = glowX + 'px';
              glow.style.top = glowY + 'px';
              window.parent.requestAnimationFrame(tick);
            }}
            doc.addEventListener('mousemove', onMove);
            doc.addEventListener('mouseleave', onLeave);
            window.parent.requestAnimationFrame(tick);
          }} catch (err) {{
            /* silent */
          }}
        }})();
        </script>
        """,
        height=0,
    )


def apply_theme(theme_mode: str) -> None:
    """Apply the selected UI skin with animations and a cursor glow."""
    colors = _get_palette(theme_mode)

    st.markdown(
        f"""
        <style>
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translate3d(0, 12px, 0); }}
            to   {{ opacity: 1; transform: translate3d(0, 0, 0); }}
        }}
        @keyframes floatShimmer {{
            0%   {{ background-position: 0% 50%; }}
            50%  {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        @keyframes pulseGlow {{
            0%   {{ box-shadow: 0 0 0 0 {colors["glow"]}; }}
            70%  {{ box-shadow: 0 0 0 16px rgba(0,0,0,0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(0,0,0,0); }}
        }}
        :root {{
            --accent: {colors["accent"]};
            --accent-soft: {colors["accent_soft"]};
            --highlight: {colors["highlight"]};
            --surface: {colors["surface"]};
            --surface-alt: {colors["surface_alt"]};
            --border: {colors["border"]};
            --text: {colors["text"]};
            --muted: {colors["muted"]};
            --shadow: {colors["shadow"]};
            --glow: {colors["glow"]};
        }}
        * {{
            transition: background-color 220ms ease, color 220ms ease,
                        border-color 220ms ease, box-shadow 220ms ease,
                        transform 220ms ease, filter 220ms ease;
        }}
        .block-container {{
            max-width: 1200px;
            padding-top: 2.2rem;
            padding-bottom: 2.5rem;
            animation: fadeInUp 520ms ease-out;
        }}
        .stApp, [data-testid="stAppViewContainer"] {{
            background: {colors["bg_gradient"]};
            color: {colors["text"]};
        }}
        [data-testid="stHeader"] {{
            background: transparent;
            backdrop-filter: blur(6px);
        }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {colors["surface"]} 0%, {colors["surface_alt"]} 100%);
            border-right: 1px solid {colors["border"]};
        }}
        [data-testid="stSidebarUserContent"] {{
            padding-top: 1rem;
        }}
        [data-testid="stSidebar"] * {{
            color: {colors["text"]};
        }}
        [data-testid="stAppViewContainer"] * {{
            border-color: {colors["border"]};
        }}
        h1, h2, h3, h4, h5, h6, p, li, label, span, div {{
            color: {colors["text"]};
        }}
        h1, h2, h3 {{
            letter-spacing: -0.02em;
        }}
        .stMarkdown, .stCaption {{
            color: {colors["text"]};
        }}
        .stAlert {{
            background: {colors["surface_alt"]};
            color: {colors["text"]};
            border: 1px solid {colors["border"]};
            border-radius: 16px;
            animation: fadeInUp 420ms ease-out;
        }}
        .stExpander, .stTextInput > div > div, .stSelectbox > div > div,
        .stNumberInput > div > div, .stSlider, .stFileUploader,
        .stTextArea textarea {{
            background: {colors["surface"]};
            color: {colors["text"]};
        }}
        .stTextInput > div > div, .stSelectbox > div > div {{
            border-radius: 12px;
        }}
        .stTextInput > div > div:focus-within,
        .stSelectbox > div > div:focus-within {{
            box-shadow: 0 0 0 3px {colors["glow"]};
            border-color: {colors["accent"]} !important;
        }}
        [data-testid="stVerticalBlockBorderWrapper"] {{
            border-radius: 20px;
            background: {colors["surface"]};
            border: 1px solid {colors["border"]};
            box-shadow: 0 14px 36px {colors["shadow"]};
            animation: fadeInUp 520ms ease-out both;
        }}
        [data-testid="stVerticalBlockBorderWrapper"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 22px 48px {colors["shadow"]};
            border-color: {colors["accent_soft"]};
        }}
        [data-testid="stFileUploaderDropzone"] {{
            background: {colors["surface"]};
            border: 1.5px dashed {colors["accent_soft"]};
            border-radius: 18px;
            padding: 1.1rem;
        }}
        [data-testid="stFileUploaderDropzone"] * {{
            color: {colors["text"]} !important;
        }}
        [data-testid="stFileUploaderDropzone"] small,
        [data-testid="stFileUploaderDropzoneInstructions"] small,
        [data-testid="stFileUploaderDropzoneInstructions"] span {{
            color: {colors["muted"]} !important;
        }}
        [data-testid="stFileUploaderDropzone"] svg {{
            fill: {colors["accent"]} !important;
            color: {colors["accent"]} !important;
        }}
        [data-testid="stFileUploaderDropzone"] button {{
            background: {colors["accent"]} !important;
            color: {colors["accent_text"]} !important;
            border: 1px solid {colors["accent"]} !important;
            border-radius: 999px !important;
            font-weight: 600 !important;
            padding: 0.4rem 1rem !important;
            box-shadow: 0 6px 14px {colors["shadow"]} !important;
        }}
        [data-testid="stFileUploaderDropzone"] button:hover {{
            filter: brightness(1.08) !important;
            transform: translateY(-1px);
        }}
        [data-testid="stFileUploaderDropzone"]:hover {{
            border-color: {colors["accent"]};
            box-shadow: 0 0 0 4px {colors["glow"]};
        }}
        .stButton > button, .stDownloadButton > button {{
            background: linear-gradient(135deg, {colors["accent"]} 0%, {colors["highlight"]} 100%);
            color: {colors["accent_text"]};
            border: 1px solid {colors["accent"]};
            border-radius: 999px;
            font-weight: 600;
            min-height: 2.9rem;
            padding: 0.55rem 1.15rem;
            box-shadow: 0 10px 24px {colors["shadow"]};
            background-size: 200% 200%;
            background-position: 0% 50%;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            transform: translateY(-2px) scale(1.01);
            filter: brightness(1.08);
            background-position: 100% 50%;
            animation: pulseGlow 1.2s ease-out 1;
        }}
        .stButton > button:active, .stDownloadButton > button:active {{
            transform: translateY(0) scale(0.99);
        }}
        .stButton > button {{
            width: 100%;
        }}
        [data-testid="stSlider"] [role="slider"] {{
            box-shadow: 0 0 0 4px {colors["glow"]};
        }}
        code {{
            color: {colors["text"]};
            background: {colors["surface_alt"]};
        }}
        small, [data-testid="stCaptionContainer"] {{
            color: {colors["muted"]};
        }}
        .hero-panel {{
            position: relative;
            padding: 1.9rem 2rem;
            border: 1px solid {colors["border"]};
            border-radius: 26px;
            background:
                linear-gradient(135deg, {colors["surface"]} 0%, {colors["surface_alt"]} 100%);
            box-shadow: 0 20px 44px {colors["shadow"]};
            margin-bottom: 1.1rem;
            overflow: hidden;
            animation: fadeInUp 600ms ease-out;
        }}
        .hero-panel::before {{
            content: "";
            position: absolute;
            inset: -40%;
            background: radial-gradient(ellipse at top right,
                {colors["glow"]} 0%, transparent 60%);
            pointer-events: none;
            opacity: 0.9;
        }}
        .hero-eyebrow, .section-eyebrow {{
            position: relative;
            text-transform: uppercase;
            letter-spacing: 0.14em;
            font-size: 0.74rem;
            font-weight: 700;
            color: {colors["accent"]};
            margin-bottom: 0.55rem;
        }}
        .hero-title {{
            position: relative;
            font-size: 2.45rem;
            font-weight: 800;
            line-height: 1.08;
            margin: 0;
            background: linear-gradient(90deg, {colors["text"]} 0%, {colors["accent"]} 60%, {colors["highlight"]} 100%);
            background-size: 200% auto;
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            animation: floatShimmer 10s ease-in-out infinite;
        }}
        .hero-copy {{
            position: relative;
            max-width: 760px;
            margin: 0.9rem 0 0;
            color: {colors["muted"]};
            font-size: 1.02rem;
            line-height: 1.65;
        }}
        .metric-card {{
            position: relative;
            border: 1px solid {colors["border"]};
            border-radius: 18px;
            background: {colors["surface"]};
            padding: 1rem 1.05rem;
            min-height: 130px;
            box-shadow: 0 12px 28px {colors["shadow"]};
            margin-bottom: 0.6rem;
            overflow: hidden;
            animation: fadeInUp 520ms ease-out both;
        }}
        .metric-card::after {{
            content: "";
            position: absolute;
            left: 0; top: 0;
            height: 3px;
            width: 100%;
            background: linear-gradient(90deg, {colors["accent"]}, {colors["highlight"]});
            opacity: 0.85;
        }}
        .metric-card:hover {{
            transform: translateY(-3px);
            box-shadow: 0 18px 40px {colors["shadow"]};
            border-color: {colors["accent_soft"]};
        }}
        .metric-card-label {{
            color: {colors["muted"]};
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 700;
        }}
        .metric-card-value {{
            font-size: 1.25rem;
            font-weight: 700;
            margin-top: 0.55rem;
        }}
        .metric-card-caption {{
            color: {colors["muted"]};
            font-size: 0.92rem;
            line-height: 1.55;
            margin-top: 0.45rem;
        }}
        .section-intro {{
            color: {colors["muted"]};
            margin-top: -0.3rem;
            margin-bottom: 1rem;
            line-height: 1.55;
        }}
        .empty-state {{
            border: 1px dashed {colors["border"]};
            background: {colors["surface_alt"]};
            border-radius: 18px;
            padding: 1.3rem 1.25rem;
            color: {colors["muted"]};
            line-height: 1.6;
            animation: fadeInUp 500ms ease-out;
        }}
        .search-result-title {{
            font-size: 1.06rem;
            font-weight: 700;
            color: {colors["accent"]};
        }}
        .search-result-meta {{
            color: {colors["muted"]};
            font-size: 0.92rem;
            margin-top: 0.25rem;
        }}
        .app-footer {{
            text-align: center;
            color: {colors["muted"]};
            font-size: 0.92rem;
            padding-top: 0.8rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    _inject_cursor_glow(colors)
