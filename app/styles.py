"""Clinical Cobalt visual tokens and scoped Streamlit styling."""

CLINICAL_COBALT_CSS = """
<style>
 .stApp { background: #F6F8FA; color: #17212B; }
 .block-container { max-width: 1180px; padding-top: 2.25rem; padding-bottom: 2rem; }
 html, body, [class*="css"] { font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
 .hero { padding: 0.25rem 0 1.75rem; border-bottom: 1px solid #D9E2EC; margin-bottom: 1.25rem; }
 .hero h1 { color: #17212B; font-size: 2.25rem; line-height: 1.15; letter-spacing: -0.045em; margin: 0; }
 .hero p { color: #5B6773; font-size: 1.02rem; margin: .55rem 0 0; }
 .hero .creator { color: #7A8793; font-size: .84rem; margin-top: .42rem; }
 .selector-label, .section-title { color:#005396; font-size:.78rem; letter-spacing:.1em; font-weight:700; text-transform:uppercase; margin:1.3rem 0 .55rem; }
 .clinical-card { background: #FFF; border: 1px solid #D9E2EC; border-radius: 14px; padding: 1.35rem; min-height: 100%; box-shadow: 0 1px 2px rgba(23,33,43,.035); }
 .eyebrow { color: #005396; font-size: .72rem; letter-spacing: .11em; font-weight: 700; text-transform: uppercase; margin-bottom: .55rem; }
 .result-label { color: #17212B; font-size: 1.35rem; font-weight: 700; margin: .15rem 0; }
 .result-benign { color: #08776B; } .result-malignant { color: #B23A48; }
 .probability { color: #1345E9; font-size: 3.2rem; font-weight: 750; letter-spacing: -.06em; line-height: 1; margin: 1rem 0 .25rem; }
 .muted { color: #5B6773; font-size: .9rem; } .fineprint { color: #7A8793; font-size: .82rem; }
 .metadata-card { background: #EDF4FA; border: 1px solid #D9E2EC; border-radius: 14px; padding: 1.25rem 1.35rem; margin: 1.25rem 0; }
 .version-card { background:#FFF; border:1px solid #D9E2EC; border-radius:10px; color:#17212B; min-height:100%; padding:.8rem 1rem; }
 .version-card strong { color:#005396; font-size:.88rem; }
 .version-card p { color:#17212B; font-size:.86rem; line-height:1.35; margin:.3rem 0 0; }
 .version-card span, .version-tradeoff { color:#5B6773; font-size:.8rem; }
 .version-tradeoff { margin:.65rem 0 0; line-height:1.4; }
 .model-row { display:flex; justify-content:space-between; align-items:baseline; margin-top:.8rem; color:#17212B; font-weight:600; }
 .highlight-card, .next-card { background:#FFF; border:1px solid #D9E2EC; border-radius:12px; color:#17212B; font-weight:650; margin:.45rem 0; min-height:4.8rem; padding:1rem; }
 .next-card p { color:#5B6773; font-size:.88rem; font-weight:400; margin:.4rem 0 0; }
 .disclaimer { border-left: 3px solid #005396; background:#FFF; border-radius: 8px; padding: .85rem 1rem; color:#5B6773; font-size:.86rem; margin-top:1.5rem; }
 [data-testid="stButton"] button { width:100%; background:#1345E9; color:#FFF; border:0; border-radius:8px; min-height:2.8rem; font-weight:700; letter-spacing:.02em; }
 [data-testid="stButton"] button:hover { background:#0D36B8; color:#FFF; }
 [data-testid="stFileUploader"] { background:#FFF; border:1px dashed #9AB4CE; border-radius:12px; padding:.5rem; }
 @media (max-width: 700px) { .block-container { padding: 1rem; } .hero h1 { font-size: 1.85rem; } .probability { font-size: 2.65rem; } }
</style>
"""
