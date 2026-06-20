"""
Analytics, Reporting & Dashboard (Task 7)
==========================================
Generates:
  - HTML dashboard with charts (Plotly)
  - CSV searchable records
  - PDF report (via weasyprint or fpdf2)
  - Spatial heatmap of violations
  - Time-series trend analysis
Author: Team AutoViolate | Flipkart Gridhackathon Round 2 | Theme 3
"""
from __future__ import annotations

import json
import csv
import os
import datetime
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict, Counter


# ---------------------------------------------------------------------------
# Analytics engine
# ---------------------------------------------------------------------------

class ViolationAnalytics:
    """
    Aggregates ViolationRecord lists into statistics and trends.
    """

    def __init__(self, class_names: List[str]):
        self.class_names = class_names
        self.records: List[Dict] = []

    def add_records(self, records: List[Dict]):
        self.records.extend(records)

    def summary_statistics(self) -> Dict:
        if not self.records:
            return {}
        total = len(self.records)
        by_class  = Counter(r["class_name"] for r in self.records)
        by_sev    = Counter(r.get("severity", "unknown") for r in self.records)
        total_fine = sum(r.get("fine_inr", 0) for r in self.records)
        plate_ok  = sum(1 for r in self.records if r.get("plate_valid", False))
        avg_conf  = sum(r.get("confidence", 0) for r in self.records) / total
        return {
            "total_violations": total,
            "by_class":   dict(by_class),
            "by_severity": dict(by_sev),
            "total_fine_inr": total_fine,
            "plate_recognition_rate": round(plate_ok / total, 4),
            "avg_confidence": round(avg_conf, 4),
        }

    def time_series_trends(self, bucket_minutes: int = 60) -> List[Dict]:
        """Bucket violations into hourly time slots."""
        buckets: Dict[str, int] = defaultdict(int)
        for r in self.records:
            ts = r.get("timestamp", 0)
            dt = datetime.datetime.fromtimestamp(ts)
            bucket = dt.strftime(f"%Y-%m-%d %H:{(dt.minute // bucket_minutes) * bucket_minutes:02d}")
            buckets[bucket] += 1
        return [{"time": k, "count": v} for k, v in sorted(buckets.items())]

    def top_violations(self, n: int = 10) -> List[Dict]:
        counts = Counter(r["class_name"] for r in self.records)
        return [{"class": k, "count": v} for k, v in counts.most_common(n)]

    def spatial_heatmap(self, image_size: tuple = (640, 640)) -> List[Dict]:
        """Return grid cell violation counts for heatmap rendering."""
        grid_h, grid_w = 20, 20
        cell_h = image_size[0] / grid_h
        cell_w = image_size[1] / grid_w
        grid = defaultdict(int)
        for r in self.records:
            bbox = r.get("bbox", [0, 0, 0, 0])
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            gi = min(int(cy / cell_h), grid_h - 1)
            gj = min(int(cx / cell_w), grid_w - 1)
            grid[(gi, gj)] += 1
        return [{"row": k[0], "col": k[1], "count": v} for k, v in grid.items()]

    def searchable_records(self, query_class: str = None, query_plate: str = None,
                            min_confidence: float = 0.0) -> List[Dict]:
        """Filter records by class name, plate text, and confidence."""
        results = self.records
        if query_class:
            results = [r for r in results if query_class.lower() in r.get("class_name", "").lower()]
        if query_plate:
            results = [r for r in results if query_plate.upper() in (r.get("plate_text") or "")]
        results = [r for r in results if r.get("confidence", 0) >= min_confidence]
        return results


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def export_csv(records: List[Dict], out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    if not records:
        print("[CSV] No records to export.")
        return
    fieldnames = list(records[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"[CSV] Exported {len(records)} records → {out_path}")


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------

def generate_html_dashboard(stats: Dict, trends: List[Dict],
                              top_viol: List[Dict], heatmap_data: List[Dict],
                              out_path: str, title: str = "AutoViolate Dashboard"):
    """
    Generates a self-contained HTML dashboard using Plotly CDN.
    """
    # Prepare chart data
    classes = [v["class"] for v in top_viol]
    counts  = [v["count"] for v in top_viol]
    trend_x = [t["time"] for t in trends]
    trend_y = [t["count"] for t in trends]

    sev_labels = list(stats.get("by_severity", {}).keys())
    sev_values = list(stats.get("by_severity", {}).values())

    # Heatmap grid
    grid_rows = max((d["row"] for d in heatmap_data), default=0) + 1
    grid_cols = max((d["col"] for d in heatmap_data), default=0) + 1
    heat_z = [[0] * grid_cols for _ in range(grid_rows)]
    for d in heatmap_data:
        heat_z[d["row"]][d["col"]] = d["count"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>{title}</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  :root {{
    --bg: #0f0f17; --card: #1a1a2e; --accent: #e94560;
    --green: #0f9b58; --text: #e0e0f0; --sub: #8888aa;
    --border: #2a2a4a;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--text); font-family:'Segoe UI',sans-serif; }}
  header {{ background:linear-gradient(135deg,#1a1a2e,#16213e);
            padding:2rem; text-align:center; border-bottom:2px solid var(--accent); }}
  header h1 {{ font-size:2rem; letter-spacing:2px; color:#fff; }}
  header p  {{ color:var(--sub); margin-top:.4rem; font-size:.95rem; }}
  .kpi-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr));
               gap:1rem; padding:1.5rem; }}
  .kpi {{ background:var(--card); border:1px solid var(--border); border-radius:12px;
          padding:1.2rem; text-align:center; transition:transform .2s; }}
  .kpi:hover {{ transform:translateY(-4px); }}
  .kpi .num  {{ font-size:2rem; font-weight:700; color:var(--accent); }}
  .kpi .label {{ font-size:.8rem; color:var(--sub); margin-top:.3rem; letter-spacing:1px; text-transform:uppercase; }}
  .charts {{ display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; padding:0 1.5rem 1.5rem; }}
  .chart-card {{ background:var(--card); border:1px solid var(--border);
                 border-radius:12px; padding:1rem; }}
  .chart-card h3 {{ color:var(--sub); font-size:.85rem; letter-spacing:1px;
                    text-transform:uppercase; margin-bottom:.8rem; }}
  footer {{ text-align:center; padding:1.5rem; color:var(--sub); font-size:.8rem;
            border-top:1px solid var(--border); margin-top:1rem; }}
  @media (max-width:800px) {{ .charts {{ grid-template-columns:1fr; }} }}
</style>
</head>
<body>
<header>
  <h1>🚦 AutoViolate CV Dashboard</h1>
  <p>Flipkart Gridhackathon Round 2 | Theme 3 — Automated Traffic Violation Detection</p>
</header>

<div class="kpi-grid">
  <div class="kpi"><div class="num">{stats.get('total_violations',0)}</div><div class="label">Total Violations</div></div>
  <div class="kpi"><div class="num">₹{stats.get('total_fine_inr',0):,}</div><div class="label">Total Fines (INR)</div></div>
  <div class="kpi"><div class="num">{stats.get('avg_confidence',0)*100:.1f}%</div><div class="label">Avg Confidence</div></div>
  <div class="kpi"><div class="num">{stats.get('plate_recognition_rate',0)*100:.1f}%</div><div class="label">Plate Recognition Rate</div></div>
</div>

<div class="charts">
  <div class="chart-card">
    <h3>Violations by Type</h3>
    <div id="bar-chart" style="height:320px;"></div>
  </div>
  <div class="chart-card">
    <h3>Severity Distribution</h3>
    <div id="pie-chart" style="height:320px;"></div>
  </div>
  <div class="chart-card">
    <h3>Violation Trend (Hourly)</h3>
    <div id="trend-chart" style="height:320px;"></div>
  </div>
  <div class="chart-card">
    <h3>Spatial Heatmap</h3>
    <div id="heat-chart" style="height:320px;"></div>
  </div>
</div>

<footer>Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | AutoViolate-CV v1.0.0</footer>

<script>
const LAYOUT = {{
  paper_bgcolor:'transparent', plot_bgcolor:'transparent',
  font:{{color:'#e0e0f0',size:11}},
  margin:{{l:40,r:10,t:10,b:60}},
  xaxis:{{gridcolor:'#2a2a4a',tickfont:{{size:10}}}},
  yaxis:{{gridcolor:'#2a2a4a'}}
}};
const CONFIG = {{displayModeBar:false,responsive:true}};

// Bar chart
Plotly.newPlot('bar-chart', [{{
  type:'bar', x:{json.dumps(classes)}, y:{json.dumps(counts)},
  marker:{{color:'#e94560',opacity:0.85}},
  text:{json.dumps(counts)}, textposition:'outside'
}}], {{...LAYOUT,xaxis:{{...LAYOUT.xaxis,tickangle:-30}}}}, CONFIG);

// Pie chart
const SEV_COLOURS = {{'critical':'#e94560','high':'#ff7043','medium':'#ffca28','low':'#0f9b58','info':'#90a4ae','unknown':'#607d8b'}};
Plotly.newPlot('pie-chart', [{{
  type:'pie', labels:{json.dumps(sev_labels)}, values:{json.dumps(sev_values)},
  hole:0.45, marker:{{colors:{json.dumps(sev_labels)}.map(l=>SEV_COLOURS[l]||'#607d8b')}},
  textinfo:'label+percent', textfont:{{size:11}}
}}], {{...LAYOUT,margin:{{l:10,r:10,t:10,b:10}}}}, CONFIG);

// Trend line
Plotly.newPlot('trend-chart', [{{
  type:'scatter', mode:'lines+markers',
  x:{json.dumps(trend_x)}, y:{json.dumps(trend_y)},
  line:{{color:'#00d4ff',width:2}},
  marker:{{color:'#00d4ff',size:5}},
  fill:'tozeroy', fillcolor:'rgba(0,212,255,0.1)'
}}], LAYOUT, CONFIG);

// Heatmap
Plotly.newPlot('heat-chart', [{{
  type:'heatmap', z:{json.dumps(heat_z)},
  colorscale:[['0','#1a1a2e'],['0.5','#e94560'],['1','#ff9800']],
  showscale:true
}}], {{...LAYOUT,margin:{{l:30,r:60,t:10,b:30}}}}, CONFIG);
</script>
</body>
</html>"""

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[Report] HTML dashboard → {out_path}")


# ---------------------------------------------------------------------------
# PDF report (fpdf2 fallback)
# ---------------------------------------------------------------------------

def generate_pdf_report(stats: Dict, top_viol: List[Dict], out_path: str):
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(233, 69, 96)
        pdf.cell(0, 12, "AutoViolate-CV Violation Report", ln=True, align="C")
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(50, 50, 80)
        pdf.cell(0, 8, f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
        pdf.ln(6)

        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(30, 30, 60)
        pdf.cell(0, 10, "Summary Statistics", ln=True)
        pdf.set_font("Helvetica", "", 11)
        pdf.set_text_color(60, 60, 80)
        for k, v in stats.items():
            if not isinstance(v, dict):
                pdf.cell(0, 8, f"  {k}: {v}", ln=True)
        pdf.ln(4)

        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 10, "Top Violations", ln=True)
        pdf.set_font("Helvetica", "", 11)
        for item in top_viol:
            pdf.cell(0, 8, f"  {item['class']}: {item['count']}", ln=True)

        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        pdf.output(out_path)
        print(f"[Report] PDF → {out_path}")
    except ImportError:
        print("[Report] fpdf2 not installed; skipping PDF generation.")


# ---------------------------------------------------------------------------
# Master reporter
# ---------------------------------------------------------------------------

class ReportGenerator:
    """
    Orchestrates analytics and produces all report formats.

    Usage
    -----
    rg = ReportGenerator(class_names=CLASS_NAMES, out_dir="./reports")
    rg.add_records(violation_dicts)
    rg.generate_all()
    """

    def __init__(self, class_names: List[str], out_dir: str = "./reports"):
        self.analytics   = ViolationAnalytics(class_names)
        self.out_dir     = Path(out_dir)
        self.class_names = class_names

    def add_records(self, records: List[Dict]):
        self.analytics.add_records(records)

    def generate_all(self):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        stats     = self.analytics.summary_statistics()
        trends    = self.analytics.time_series_trends()
        top_viol  = self.analytics.top_violations(10)
        heatmap   = self.analytics.spatial_heatmap()

        # Save JSON summary
        with open(self.out_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        # CSV records
        export_csv(self.analytics.records, str(self.out_dir / "violation_records.csv"))

        # HTML dashboard
        generate_html_dashboard(stats, trends, top_viol, heatmap,
                                  str(self.out_dir / "dashboard.html"))

        # PDF report
        generate_pdf_report(stats, top_viol, str(self.out_dir / "report.pdf"))

        print(f"\n[Reporter] All reports saved to {self.out_dir}")
        return stats


if __name__ == "__main__":
    import time, random
    from src.models.classifier import VIOLATION_CLASSES
    names = [VIOLATION_CLASSES[i]["name"] for i in range(8)]
    rg = ReportGenerator(names, out_dir="./sample_reports")
    # Synthetic records
    records = []
    for _ in range(200):
        cls_id = random.randint(0, 7)
        meta   = VIOLATION_CLASSES[cls_id]
        records.append({
            "image_id": f"img_{random.randint(0,999)}",
            "timestamp": time.time() - random.randint(0, 86400),
            "class_id": cls_id,
            "class_name": meta["name"],
            "confidence": round(random.uniform(0.5, 0.99), 3),
            "severity": meta["severity"],
            "fine_inr": meta["fine_inr"],
            "bbox": [100, 100, 400, 400],
            "plate_text": f"MH{random.randint(10,99)}AB{random.randint(1000,9999)}" if random.random() > 0.4 else None,
            "plate_valid": random.random() > 0.3,
        })
    rg.add_records(records)
    stats = rg.generate_all()
    print("Stats:", stats)
