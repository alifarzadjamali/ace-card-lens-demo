"""Shared styling and page components for AceVision."""
import streamlit as st


def apply_style():
    st.markdown("""
<style>
:root { --gold:#D5B26C; --ink:#0D1017; --muted:#A6ADBA; }
.stApp { background:radial-gradient(ellipse at 90% 0%, #252630 0, #0D1017 55%); color:#F5F3EE; }
html, body, [class*="css"], h1, h2, h3 {font-family:Arial, sans-serif;}
.cl-hero *, .cl-masthead *, .cl-section *, .cl-empty *, .cl-footer, .cl-step, .cl-panel-label *, .cl-card-meta {font-family:Arial, sans-serif;}
.block-container {max-width:1360px; padding:2rem 3rem 4rem;}
header[data-testid="stHeader"] {background:transparent;}
[data-testid="stSidebar"] {background:#141922;}
.cl-masthead {display:flex; align-items:center; justify-content:space-between; border-bottom:1px solid #34353B; padding:0 0 20px; margin-bottom:24px;}
.cl-brand {font-size:23px; font-weight:700; letter-spacing:-.8px;}
.cl-brand b {color:var(--gold);}
.cl-brand small {display:inline-block; color:var(--muted); font-size:10px; letter-spacing:2px; margin-left:18px; font-weight:400;}
.cl-live {font-size:10px; letter-spacing:1.6px; color:#D5B26C; border:1px solid #514735; padding:8px 12px; border-radius:30px;}
.cl-hero {position:relative; overflow:hidden; display:grid; grid-template-columns:1.5fr 1fr; align-items:center; gap:30px; padding:28px 38px; margin-bottom:26px; border:1px solid #3A3B40; border-radius:18px; background:linear-gradient(115deg,#181D27,#10141C 60%,#292720);}
.cl-eyebrow {color:var(--gold); font-size:10px; font-weight:700; letter-spacing:2.2px; text-transform:uppercase; margin-bottom:14px;}
.cl-hero h1 {font-size:clamp(30px,3.3vw,46px); line-height:1.12; letter-spacing:-1.8px; font-weight:700; padding:0; margin:0 0 14px;}
.cl-hero h1 em {font-style:normal; color:var(--gold);}
.cl-hero p {color:#ADB4C0; font-size:14px; max-width:470px; line-height:1.65; margin:0;}
.cl-facts {display:flex; gap:25px; margin-top:22px; font-size:11px; color:var(--muted);}
.cl-facts strong {color:#F5F3EE; font-size:16px; margin-right:5px;}
.cl-visual {position:relative; height:235px; display:flex; align-items:center; justify-content:center; background:radial-gradient(ellipse,#D5B26C12,transparent 70%);}
.cl-slab {width:137px; height:194px; transform:rotate(9deg); border:1px solid #BDA16D; border-radius:15px; padding:10px; background:linear-gradient(130deg,#494B4C,#191E27 40%,#3A3A36); box-shadow:12px 18px 45px #0008;}
.cl-slab-label {font-size:8px; letter-spacing:1.6px; border:1px solid #8F7C55; padding:8px; color:#E9D3A0; border-radius:5px; margin-bottom:9px;}
.cl-slab-art {height:130px; position:relative; overflow:hidden; border:4px solid #CDBB8D; border-radius:6px; background:linear-gradient(145deg,#456979,#111C2B 55%,#A68C59);}
.cl-slab-art:before {content:""; position:absolute; width:95px; height:95px; background:linear-gradient(135deg,#DFC27D,#365767); transform:rotate(45deg); top:60px; left:25px;}
.cl-slab-art:after {content:"◇"; position:absolute; top:2px; left:30px; font-size:64px; color:#F7DF96; text-shadow:0 0 20px #E9D3A099;}
.cl-scan {position:absolute; width:210px; height:211px; border:1px solid #D5B26C55; border-radius:12px; transform:rotate(-4deg);}
.cl-scan:after {content:""; position:absolute; top:53%; left:-15px; right:-15px; height:1px; background:#E9D3A0; box-shadow:0 0 18px #D5B26C;}
.cl-callout {position:absolute; font-size:9px; letter-spacing:1.3px; color:#E9D3A0; background:#141922; border:1px solid #514735; border-radius:5px; padding:7px 10px;}
.cl-callout.one {left:0; top:35px;}
.cl-callout.two {right:0; bottom:33px;}
.cl-section {display:flex; align-items:center; gap:13px; margin:24px 0 16px;}
.cl-section-num {font-size:11px; color:var(--gold); border:1px solid #554B38; border-radius:8px; padding:9px;}
.cl-section h2 {font-size:20px; padding:0; margin:0; letter-spacing:-.5px;}
.cl-section p {font-size:12px; color:var(--muted); margin:4px 0 0;}
[data-baseweb="tab-list"] {gap:24px; border-bottom:1px solid #303640; background:transparent;}
[data-baseweb="tab"] {font-size:13px; padding:8px 0 12px;}
[data-baseweb="tab"][aria-selected="true"] {color:var(--gold);}
[data-baseweb="tab-highlight"] {background:var(--gold);}
[data-testid="stBaseButton-primary"] {background:var(--gold); border:1px solid var(--gold); color:#15171B; min-height:45px; border-radius:9px; font-weight:700;}
[data-testid="stBaseButton-primary"]:hover {background:#E9D3A0; border-color:#E9D3A0; color:#15171B;}
[data-testid="stMetric"] {height:142px; background:linear-gradient(135deg,#1C222D,#141922); border:1px solid #323945; border-radius:12px; padding:18px 20px; border-top:2px solid #9D8454;}
[data-testid="stMetricLabel"] {color:#ADB4C0; font-size:12px;}
[data-testid="stMetricValue"] {font-size:26px; letter-spacing:-.7px;}
.cl-panel-label {border:1px solid #343A45; border-bottom:0; border-radius:12px 12px 0 0; padding:14px; margin-top:18px; background:#191F29; font-size:13px;}
.cl-panel-label span {display:block; font-size:9px; letter-spacing:1.8px; color:var(--gold); margin-bottom:5px;}
[class*="st-key-inspection_"] {border:1px solid #343A45; border-radius:12px; overflow:hidden; background:#141922; margin-top:18px; gap:0;}
[class*="st-key-inspection_"] .cl-panel-label {border:0; border-bottom:1px solid #343A45; border-radius:0; margin:0;}
[class*="st-key-inspection_"] [data-testid="stElementContainer"] {width:100% !important;}
[class*="st-key-inspection_"] div:has(> [data-testid="stImage"]) {width:100% !important;}
[class*="st-key-inspection_"] [data-testid="stImage"] {width:100% !important; max-width:none; padding:18px 16px 12px; box-sizing:border-box; display:flex; justify-content:center;}
[class*="st-key-inspection_"] [data-testid="stImageContainer"] {width:100% !important; max-width:none; display:flex; flex-direction:column; align-items:center;}
[class*="st-key-inspection_"] [data-testid="stImage"] img {display:block; width:100% !important; height:390px; object-fit:contain; object-position:center;}
[class*="st-key-inspection_"] [data-testid="stImageCaption"] {width:100% !important; min-height:64px; padding-top:12px; text-align:center; box-sizing:border-box;}
[data-testid="stImage"] img {border-radius:9px; object-fit:contain; max-height:390px;}
[data-testid="stImageCaption"] {font-size:11px; color:#A6ADBA;}
.cl-step {display:inline-block; padding:6px 11px; margin:3px 5px 3px 0; border-radius:6px; border:1px solid #3D4552; font-size:11px; color:#BEC5D0;}
.cl-step.done {background:#D5B26C12; border-color:#D5B26C44; color:#E9D3A0;}
.cl-card-meta {font-size:12px; line-height:1.7; color:#C1C6D0;}
.cl-small {font-size:10px; color:#9CA5B3;}
.cl-empty {display:grid; grid-template-columns:repeat(3,1fr); gap:16px; margin:24px 0;}
.cl-empty article {border:1px solid #303744; border-radius:12px; padding:22px; background:#151A23;}
.cl-empty span {color:var(--gold); font-size:10px; letter-spacing:2px;}
.cl-empty h3 {font-size:16px !important; letter-spacing:0 !important; color:#F5F3EE !important; margin:12px 0 8px; padding:0;}
.cl-empty p {font-size:12px; line-height:1.6; color:#A6ADBA; margin:0;}
.cl-footer {border-top:1px solid #303640; padding-top:16px; color:#929BA9; font-size:11px; line-height:1.8;}
@media(max-width:750px) {
 .block-container {padding:1rem;}
 .cl-hero {grid-template-columns:1fr; padding:24px;}
 .cl-visual {display:none;}
 .cl-brand small {display:none;}
 .cl-empty {grid-template-columns:1fr;}
 .cl-facts {gap:12px; flex-wrap:wrap;}
}
</style>""", unsafe_allow_html=True)


def hero(cards, sets):
    st.markdown(f"""
<div class="cl-masthead">
 <div class="cl-brand"><b>◇</b> AceVision <small>INSPECTION STUDIO</small></div>
 <div class="cl-live">INTERACTIVE DEMO</div>
</div>
<div class="cl-hero">
 <div>
  <h1>Let's take a closer look at<br><em>what AceVision could look like.</em></h1>
  <p>Try a card below and see what the app picks up. You can check the results and decide what needs another look.</p>
  <div class="cl-facts"><div><strong>{cards}</strong> cards</div><div><strong>{sets}</strong> sets</div></div>
 </div>
 <div class="cl-visual" aria-label="Illustrative card scanning graphic">
  <div class="cl-scan"></div><div class="cl-slab"><div class="cl-slab-label">ACEVISION / CONCEPT</div><div class="cl-slab-art"></div></div>
  <div class="cl-callout one">01 / IDENTIFY</div><div class="cl-callout two">02 / INSPECT</div>
 </div>
</div>""", unsafe_allow_html=True)


def section(number, title, description):
    st.markdown(f'<div class="cl-section"><div class="cl-section-num">{number}</div>'
                f'<div><h2>{title}</h2><p>{description}</p></div></div>', unsafe_allow_html=True)


def empty_state():
    st.markdown("""
<div class="cl-empty">
 <article><span>IDENTIFY</span><h3>Find your card</h3><p>Compare your photo with the catalogue. Check the printed details to see which match is right.</p></article>
 <article><span>INSPECT</span><h3>Check its condition</h3><p>Look at the borders and marked areas for possible wear. The surface view shows differences from the reference card.</p></article>
 <article><span>REVIEW</span><h3>Add your review</h3><p>Decide which findings you agree with, correct the card match if needed, and download your notes.</p></article>
</div>
<div class="cl-footer">Built by Ali (Farzad) Jamali · Independent concept for the Ace Grading / University of Salford KTP interview.<br>
Photo-based condition signals for human review. This demo does not issue a final grade.</div>
""", unsafe_allow_html=True)
