# -*- coding: utf-8 -*-
"""Render slides_data.py into a single self-contained HTML deck.

    python presentation/build_deck.py

Output: presentation/ANSHUMAN_deck.html  -- no network, no build step, opens
from a file:// URL, so it works with the venue wifi down.

Keys:  \u2192 / space  next \u00b7  \u2190  back \u00b7  N  speaker notes \u00b7  F  full screen
       G  grid overview \u00b7  Home  first \u00b7  End  last \u00b7  0-9  jump to act
"""
import html
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from slides_data import S, SECTIONS, TITLE, SUBTITLE, META          # noqa: E402

OUT = os.path.join(HERE, "ANSHUMAN_deck.html")

CSS = """
:root{--bg:#0a0e14;--bg2:#0f141c;--ink:#e6edf3;--dim:#8b98a5;--acc:#3fb950;
      --acc2:#58a6ff;--warn:#f0883e;--bad:#f85149;--line:#1f2933;}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:var(--bg);color:var(--ink);
  font-family:"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;overflow:hidden;height:100%}
#deck{position:fixed;inset:0;display:flex;flex-direction:column}
.slide{display:none;flex:1;padding:3.2vh 4vw 9vh;overflow:auto;animation:fade .25s ease}
.slide.on{display:flex;flex-direction:column}
@keyframes fade{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
h1{font-size:clamp(28px,4.6vw,64px);line-height:1.08;margin:0 0 .3em;letter-spacing:-.02em;font-weight:700}
h2{font-size:clamp(22px,2.9vw,40px);line-height:1.15;margin:0 0 .25em;font-weight:700}
.sub{color:var(--dim);font-size:clamp(14px,1.5vw,22px);line-height:1.45;margin:0 0 1.6em;max-width:60ch}
ul{margin:0;padding:0;list-style:none;font-size:clamp(15px,1.55vw,23px);line-height:1.5}
li{margin:0 0 .55em;padding-left:1.5em;position:relative;max-width:78ch}
li:before{content:"\\25B8";color:var(--acc);position:absolute;left:0;top:0}
li.call{background:linear-gradient(90deg,rgba(63,185,80,.14),transparent 70%);
  border-left:3px solid var(--acc);padding:.5em .8em .5em 1.1em;margin:.9em 0;
  font-weight:600;max-width:none}
li.call:before{content:"";left:-3px}
li.sub2{margin-left:1.6em;font-size:.86em;color:#c3ced9}
li.sub2:before{content:"\\2013";color:var(--dim)}
li code,p code,td code{background:#161b22;border:1px solid var(--line);border-radius:4px;
  padding:.08em .35em;font-family:"Cascadia Code",Consolas,monospace;font-size:.86em;color:#79c0ff}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px;margin-top:.4em}
.card{background:var(--bg2);border:1px solid var(--line);border-left:3px solid var(--acc);
  border-radius:8px;padding:18px 20px}
.card .v{font-size:clamp(24px,3vw,42px);font-weight:700;color:var(--acc);line-height:1;
  font-variant-numeric:tabular-nums}
.card .l{color:var(--dim);font-size:clamp(11px,1.05vw,15px);line-height:1.4;margin-top:9px;white-space:pre-line}
table{border-collapse:collapse;width:100%;font-size:clamp(12px,1.25vw,18px);margin-top:.3em}
th,td{border-bottom:1px solid var(--line);padding:.45em .6em;text-align:left;vertical-align:top}
th{color:var(--acc2);font-weight:600;border-bottom:2px solid var(--line);white-space:nowrap}
tr:nth-child(even) td{background:rgba(255,255,255,.018)}
tr.ours td{background:rgba(63,185,80,.11);font-weight:600}
.quote{font-size:clamp(22px,3.1vw,44px);line-height:1.35;font-weight:600;max-width:36ch}
.quote div{margin:0 0 .6em}
.quote div:nth-child(1){color:var(--acc2)}
.quote div:nth-child(2){color:var(--acc)}
.quote div:nth-child(3){color:var(--dim);font-size:.5em;font-weight:400;max-width:52ch;margin-top:1.4em}
.tag{display:inline-block;font-size:12px;letter-spacing:.14em;text-transform:uppercase;
  color:var(--acc);border:1px solid var(--acc);border-radius:20px;padding:3px 12px;margin-bottom:1.2em}
.sectnum{font-size:clamp(60px,12vw,180px);color:var(--line);font-weight:800;line-height:.9}
#bar{position:fixed;left:0;bottom:0;height:3px;background:var(--acc);width:0;transition:width .2s}
#hud{position:fixed;right:16px;bottom:12px;color:var(--dim);font-size:12px;
  font-family:Consolas,monospace;letter-spacing:.04em;text-align:right}
#hud b{color:var(--ink)}
#notes{position:fixed;left:0;right:0;bottom:0;display:none;background:rgba(4,7,11,.97);
  border-top:1px solid var(--acc);padding:16px 4vw 20px;max-height:46vh;overflow:auto;z-index:9}
#notes.on{display:block}
#notes .k{color:var(--acc);font-size:11px;letter-spacing:.14em;text-transform:uppercase;margin-bottom:8px}
#notes p{margin:0 0 10px;font-size:15px;line-height:1.55;color:#dbe4ec;max-width:110ch}
#notes .kid{color:var(--warn);border-left:3px solid var(--warn);padding-left:10px;font-style:italic}
#grid{position:fixed;inset:0;background:rgba(4,7,11,.98);display:none;overflow:auto;
  padding:3vh 3vw;z-index:20}
#grid.on{display:block}
#g{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:10px}
.gi{background:var(--bg2);border:1px solid var(--line);border-radius:6px;padding:10px 12px;
  font-size:12px;line-height:1.35;cursor:pointer}
.gi:hover{border-color:var(--acc)}
.gi .n{color:var(--acc);font-family:Consolas,monospace}
.gi.cur{border-color:var(--acc);background:rgba(63,185,80,.12)}
.meta{color:var(--dim);font-size:clamp(12px,1.2vw,17px);margin-top:auto;padding-top:2em}
"""

JS = r"""
const SLIDES=[...document.querySelectorAll('.slide')];let i=+(location.hash.slice(1)||0);
if(i<0||i>=SLIDES.length)i=0;
function show(n){i=Math.max(0,Math.min(SLIDES.length-1,n));
 SLIDES.forEach((s,k)=>s.classList.toggle('on',k===i));
 document.getElementById('bar').style.width=((i+1)/SLIDES.length*100)+'%';
 const t=SLIDES[i].dataset.t||'';
 document.getElementById('hud').innerHTML='<b>'+(i+1)+'</b>/'+SLIDES.length+
   ' &nbsp; '+(t?'\u23F1 '+t+' min &nbsp; ':'')+
   (SLIDES[i].dataset.sec?'act '+SLIDES[i].dataset.sec:'');
 document.getElementById('notes').classList.remove('on');
 document.querySelectorAll('#g .gi').forEach((e,k)=>e.classList.toggle('cur',k===i));
 location.hash=i;SLIDES[i].scrollTop=0;}
function notes(){document.getElementById('notes').classList.toggle('on');
 document.getElementById('notes').innerHTML=
  '<div class="k">Speaker notes \u2014 slide '+(i+1)+'</div><p>'+(SLIDES[i].dataset.notes||'')+'</p>'+
  (SLIDES[i].dataset.kid?'<p class="kid">10-year-old: '+SLIDES[i].dataset.kid+'</p>':'');}
function grid(){document.getElementById('grid').classList.toggle('on');}
addEventListener('keydown',e=>{
 if(e.key==='ArrowRight'||e.key===' '||e.key==='PageDown'){show(i+1);e.preventDefault();}
 else if(e.key==='ArrowLeft'||e.key==='PageUp'){show(i-1);e.preventDefault();}
 else if(e.key==='n'||e.key==='N')notes();
 else if(e.key==='g'||e.key==='G')grid();
 else if(e.key==='f'||e.key==='F'){document.fullscreenElement?document.exitFullscreen():document.documentElement.requestFullscreen();}
 else if(e.key==='Home')show(0);
 else if(e.key==='End')show(SLIDES.length-1);
 else if(/^[0-9]$/.test(e.key)){const j=SLIDES.findIndex(s=>s.dataset.sec===e.key);if(j>=0)show(j);}
});
addEventListener('click',e=>{if(e.target.closest('#grid')&&!e.target.closest('.gi'))grid();});
const g=document.getElementById('g');
SLIDES.forEach((s,k)=>{const d=document.createElement('div');d.className='gi';
 d.innerHTML='<span class="n">'+String(k+1).padStart(2,'0')+'</span> '+(s.dataset.title||'');
 d.onclick=()=>{show(k);grid();};g.appendChild(d);});
show(i);
""".replace("(SLIDES[i].dataset.title||'')", "(SLIDES[i].querySelector('h1,h2')?.textContent||'')")


def esc(x):
    return html.escape(str(x), quote=False)


def inline(x):
    """escape, then restore `code` and >>/-- markers handled by the caller"""
    return esc(x).replace("`", "")


def code_fmt(x):
    out = esc(x)
    parts = out.split("`")
    for k in range(1, len(parts), 2):
        parts[k] = "<code>" + parts[k] + "</code>"
    return "".join(parts)


def bullets_html(b):
    out = ["<ul>"]
    for line in b:
        if line.startswith(">>"):
            out.append(f'<li class="call">{code_fmt(line[2:].strip())}</li>')
        elif line.startswith("--"):
            out.append(f'<li class="sub2">{code_fmt(line[2:].strip())}</li>')
        else:
            out.append(f"<li>{code_fmt(line)}</li>")
    out.append("</ul>")
    return "".join(out)


def table_html(rows):
    out = ["<table>"]
    for r, row in enumerate(rows):
        cls = ' class="ours"' if r and ("ours" in str(row[0]).lower()
                                       or "this work" in str(row[0]).lower()) else ""
        tag = "th" if r == 0 else "td"
        out.append(f"<tr{cls}>" + "".join(
            f"<{tag}>{code_fmt(c)}</{tag}>" for c in row) + "</tr>")
    out.append("</table>")
    return "".join(out)


def render(s):
    kind = s.get("kind", "bullets")
    data = (f'data-sec="{s.get("sec","")}" data-t="{s.get("t","")}" '
            f'data-notes="{esc(s.get("notes",""))}" data-kid="{esc(s.get("kid",""))}"')
    if kind == "title":
        body = (f'<div class="tag">M.Tech Dissertation \u00b7 Defence Review</div>'
                f"<h1>{esc(s['title'])}</h1><p class='sub'>{esc(s['sub'])}</p>"
                f"<p class='meta'>{esc(s.get('meta',''))}</p>")
    elif kind == "section":
        body = (f'<div class="sectnum">{esc(s.get("sec",""))}</div>'
                f"<h1>{esc(s['title'])}</h1><p class='sub'>{esc(s.get('sub',''))}</p>")
    elif kind == "metric":
        cards = "".join(f'<div class="card"><div class="v">{esc(v)}</div>'
                        f'<div class="l">{esc(l)}</div></div>' for v, l in s["cards"])
        body = f"<h2>{esc(s['title'])}</h2><p class='sub'>{esc(s.get('sub',''))}</p><div class='cards'>{cards}</div>"
    elif kind == "table":
        body = (f"<h2>{esc(s['title'])}</h2><p class='sub'>{code_fmt(s.get('sub',''))}</p>"
                + table_html(s["rows"]))
    elif kind == "quote":
        body = (f"<h2>{esc(s['title'])}</h2><div class='quote'>"
                + "".join(f"<div>{esc(c)}</div>" for c in s["rows"][0]) + "</div>")
    elif kind == "demo":
        body = (f"<div class='tag'>LIVE</div><h2>{esc(s['title'])}</h2>"
                f"<p class='sub'>{esc(s.get('sub',''))}</p>" + bullets_html(s["bullets"]))
    elif kind == "end":
        body = (f"<h1>{esc(s['title'])}</h1><p class='sub'>{esc(s.get('sub',''))}</p>")
    else:
        body = (f"<h2>{esc(s['title'])}</h2><p class='sub'>{code_fmt(s.get('sub',''))}</p>"
                + bullets_html(s.get("bullets", [])))
    return f'<section class="slide" {data}>{body}</section>'


def main():
    secs = {k: v for k, v in SECTIONS}
    nav = " \u00b7 ".join(f"{k} {v}" for k, v in SECTIONS)
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(TITLE)} \u2014 LPI-CGAN covert link</title><style>{CSS}</style></head><body>
<div id="deck">{''.join(render(s) for s in S)}</div>
<div id="bar"></div><div id="hud"></div>
<div id="notes"></div><div id="grid"><div id="g"></div></div>
<script>{JS}</script>
<div style="position:fixed;left:16px;bottom:12px;color:#4b5563;font-size:11px;max-width:44vw">
\u2192 next \u00b7 \u2190 back \u00b7 N notes \u00b7 G grid \u00b7 F full screen \u00b7 0-9 jump to act<br>{esc(nav)}</div>
</body></html>"""
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(doc)
    n = len(S)
    mins = sum(float(s.get("t", 2)) for s in S)
    print(f"[deck] {OUT}")
    print(f"[deck] {n} slides, {mins:.0f} min of stage time, "
          f"{os.path.getsize(OUT)//1024} KB, no external assets")
    print(f"[deck] acts: {len(SECTIONS)} | notes on {sum(1 for s in S if s.get('notes'))} slides "
          f"| kid-analogy on {sum(1 for s in S if s.get('kid'))} slides")


if __name__ == "__main__":
    main()
