"""
Builds a self-contained HTML report. Used by explore.py and
credit_card_forecast.py when --html is passed.

Self-contained means exactly that: charts are embedded in the file as images
and the styling is inline, so the report is ONE file that opens in any browser
with no internet and nothing else alongside it. That matters here -- the machine
this runs on has no internet, and the numbers are not something to upload
anywhere.

Nothing in here forecasts or analyses. It only formats.
"""

import base64
import datetime as _dt
import html as _html
import io

CSS = """
:root{
  --ink:#1a1d21; --muted:#5b6570; --line:#e3e7ec; --bg:#ffffff;
  --panel:#f7f9fb; --accent:#1f5f8b; --warm:#b4451f; --good:#1f7a4d;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:48px 32px 96px}
header{border-bottom:3px solid var(--ink);padding-bottom:20px;margin-bottom:8px}
h1{font-size:30px;line-height:1.2;margin:0 0 6px;letter-spacing:-.02em}
.sub{color:var(--muted);font-size:15px;margin:0}
.meta{color:var(--muted);font-size:13px;margin-top:14px}
h2{font-size:21px;margin:52px 0 4px;letter-spacing:-.01em;
  padding-top:20px;border-top:1px solid var(--line)}
h2 .num{color:var(--muted);font-weight:400;margin-right:10px}
h3{font-size:16px;margin:28px 0 8px;color:var(--ink)}
p{margin:10px 0;max-width:74ch}
p.lead{color:var(--muted)}
.note{background:var(--panel);border-left:3px solid var(--accent);
  padding:12px 16px;margin:16px 0;font-size:14px;max-width:74ch}
.note.warn{border-left-color:var(--warm)}
.note strong{color:var(--warm)}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:14px;
  font-variant-numeric:tabular-nums}
th{text-align:left;font-weight:600;padding:8px 12px;border-bottom:2px solid var(--ink);
  white-space:nowrap}
td{padding:7px 12px;border-bottom:1px solid var(--line)}
td.n,th.n{text-align:right}
tbody tr:hover{background:var(--panel)}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));
  gap:2px;margin:18px 0;background:var(--line);border:1px solid var(--line)}
.kv div{background:var(--bg);padding:14px 16px}
.kv .k{color:var(--muted);font-size:12px;text-transform:uppercase;
  letter-spacing:.06em;margin-bottom:4px}
.kv .v{font-size:19px;font-variant-numeric:tabular-nums}
figure{margin:22px 0}
figure img{width:100%;height:auto;display:block;border:1px solid var(--line)}
figcaption{color:var(--muted);font-size:13px;margin-top:8px;max-width:74ch}
.pos{color:var(--good)} .neg{color:var(--warm)}
.tag{display:inline-block;font-size:11px;padding:2px 7px;border-radius:3px;
  background:var(--panel);color:var(--muted);border:1px solid var(--line)}
.tag.yes{background:#e8f5ee;color:var(--good);border-color:#bfe3ce}
.tag.no{background:#fdecec;color:var(--warm);border-color:#f5cfc7}
footer{margin-top:64px;padding-top:18px;border-top:1px solid var(--line);
  color:var(--muted);font-size:13px}
@media print{
  .wrap{max-width:none;padding:0}
  h2{page-break-after:avoid} figure{page-break-inside:avoid}
  tbody tr:hover{background:none}
}
"""


def _esc(text):
  return _html.escape(str(text))


class Report:
  """Collects sections, then writes one HTML file."""

  def __init__(self, title, subtitle=""):
    self.title = title
    self.subtitle = subtitle
    self.parts = []
    self.sections = 0

  # -- structure --------------------------------------------------------------
  def h2(self, text, numbered=True):
    self.sections += 1
    num = f'<span class="num">{self.sections}</span>' if numbered else ""
    self.parts.append(f"<h2>{num}{_esc(text)}</h2>")
    return self

  def h3(self, text):
    self.parts.append(f"<h3>{_esc(text)}</h3>")
    return self

  def p(self, text, lead=False):
    cls = ' class="lead"' if lead else ""
    self.parts.append(f"<p{cls}>{_esc(text)}</p>")
    return self

  def note(self, text, warn=False):
    cls = "note warn" if warn else "note"
    self.parts.append(f'<div class="{cls}">{_esc(text)}</div>')
    return self

  def stats(self, pairs):
    """A row of headline figures."""
    cells = "".join(f'<div><div class="k">{_esc(k)}</div>'
                    f'<div class="v">{_esc(v)}</div></div>' for k, v in pairs)
    self.parts.append(f'<div class="kv">{cells}</div>')
    return self

  def table(self, headers, rows, numeric=None):
    """`numeric` is the set of column indexes to right-align."""
    numeric = numeric or set()
    head = "".join(f'<th class="n">{_esc(h)}</th>' if i in numeric
                   else f"<th>{_esc(h)}</th>" for i, h in enumerate(headers))
    body = []
    for row in rows:
      cells = []
      for i, cell in enumerate(row):
        # A cell may arrive as (text, css_class) to colour or tag it.
        if isinstance(cell, tuple):
          value, cls = cell
          cls = f' class="{cls}"' if cls else ""
          if i in numeric:
            cls = f' class="n {cell[1]}"' if cell[1] else ' class="n"'
          cells.append(f"<td{cls}>{value}</td>")
        else:
          cls = ' class="n"' if i in numeric else ""
          cells.append(f"<td{cls}>{_esc(cell)}</td>")
      body.append("<tr>" + "".join(cells) + "</tr>")
    self.parts.append(f"<table><thead><tr>{head}</tr></thead>"
                      f"<tbody>{''.join(body)}</tbody></table>")
    return self

  def figure(self, fig, caption="", close=True):
    """Embed a matplotlib figure as an image inside the file itself."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor="white")
    if close:
      import matplotlib.pyplot as plt
      plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    cap = f"<figcaption>{_esc(caption)}</figcaption>" if caption else ""
    self.parts.append(f'<figure><img alt="{_esc(caption)}" '
                      f'src="data:image/png;base64,{data}">{cap}</figure>')
    return self

  # -- output -----------------------------------------------------------------
  def save(self, path):
    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(self.title)}</title><style>{CSS}</style></head>
<body><div class="wrap">
<header>
  <h1>{_esc(self.title)}</h1>
  <p class="sub">{_esc(self.subtitle)}</p>
  <p class="meta">Generated {stamp} &middot; self-contained, no internet needed</p>
</header>
{''.join(self.parts)}
<footer>Produced locally. The data in this file never left this machine.</footer>
</div></body></html>"""
    with open(path, "w", encoding="utf-8") as fh:
      fh.write(doc)
    return path


# ------------------------------------------------------------------ charts --
PALETTE = {"line": "#1f5f8b", "warm": "#b4451f", "good": "#1f7a4d",
           "grey": "#8a949e", "grid": "#e3e7ec"}


def style_axes(ax, title="", ylabel=""):
  ax.set_title(title, fontsize=11, loc="left", color="#1a1d21", pad=8)
  if ylabel:
    ax.set_ylabel(ylabel, fontsize=9, color="#5b6570")
  ax.grid(True, alpha=0.35, color=PALETTE["grid"], linewidth=0.8)
  ax.set_axisbelow(True)
  for side in ("top", "right"):
    ax.spines[side].set_visible(False)
  for side in ("left", "bottom"):
    ax.spines[side].set_color("#c9d1d9")
  ax.tick_params(labelsize=9, colors="#5b6570")
  return ax
