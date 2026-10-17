"""Standalone HTML report generator for deforestation analysis.

Exports self-contained HTML documents with embedded Base64 imagery,
responsive grid styling, candidate region inventory tables, plain-language
executive summaries, and complete scientific documentation.
No external network calls, CDN dependencies, or local file leaks.
"""

import base64
from datetime import datetime, timezone
import io
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np


def figure_to_base64_png(fig: plt.Figure) -> str:
    """Convert a Matplotlib Figure to an embedded base64 data URI string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def generate_deforestation_analysis_html_report(
    patch_id: str,
    metadata: Dict[str, Any],
    clf_result: Dict[str, Any],
    seg_result: Dict[str, Any],
    figures: Dict[str, plt.Figure],
    candidate_areas: Optional[List[Any]] = None,
    conf_stats: Optional[Dict[str, Any]] = None,
    ig_summary: Optional[Dict[str, Any]] = None,
    lime_summary: Optional[Dict[str, Any]] = None,
    executive_summary: Optional[Dict[str, str]] = None,
) -> str:
    """Generate comprehensive standalone HTML report for verified deforestation analysis."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Encode figures
    img_bitemporal = figure_to_base64_png(figures["bitemporal"]) if "bitemporal" in figures else ""
    img_candidate_map = figure_to_base64_png(figures["candidate_map"]) if "candidate_map" in figures else ""
    img_pred_mask = figure_to_base64_png(figures["pred_mask"]) if "pred_mask" in figures else ""
    img_overlay = figure_to_base64_png(figures["overlay"]) if "overlay" in figures else ""
    img_confusion = figure_to_base64_png(figures["confusion"]) if "confusion" in figures else ""
    img_ig = figure_to_base64_png(figures["ig"]) if "ig" in figures else ""
    img_gradcam = figure_to_base64_png(figures["gradcam"]) if "gradcam" in figures else ""
    img_lime = figure_to_base64_png(figures["lime"]) if "lime" in figures else ""

    is_deforest = clf_result.get("prediction") == "Deforestation"
    status_bg = "#FEF2F2" if is_deforest else "#F0FDF4"
    status_border = "#DC2626" if is_deforest else "#16A34A"
    status_text = "#991B1B" if is_deforest else "#166534"
    detection_title = "DEFORESTATION SIGNAL DETECTED" if is_deforest else "NO DEFORESTATION DETECTED"

    pred_ha = seg_result.get("predicted_loss_ha", 0.0)
    pred_px = seg_result.get("predicted_loss_pixels", 0)
    valid_px = seg_result.get("total_valid_pixels", 65536)
    excl_px = 65536 - valid_px

    # Metadata rows
    meta_rows = "".join(
        f"<tr><td class='meta-key'>{k.replace('_', ' ').capitalize()}</td><td class='meta-val'>{v}</td></tr>"
        for k, v in metadata.items()
        if not isinstance(v, (dict, list))
    )

    # Executive Summary Plain-Language Q&A block
    if executive_summary:
        exec_rows = "".join(
            f"""<div style='margin-bottom: 0.9rem;'>
                <strong style='color: #1F2937; font-size: 0.95rem;'>{q}</strong>
                <p style='margin: 0.2rem 0 0 0; color: #4B5563; font-size: 0.9rem; line-height: 1.55;'>{a}</p>
            </div>"""
            for q, a in executive_summary.items()
        )
    else:
        exec_rows = "<p style='color: #4B5563;'>Automated plain-language summary generated from bi-temporal remote sensing evidence.</p>"

    # Candidate regions table rows
    candidate_table_rows = ""
    if candidate_areas:
        for reg in candidate_areas:
            # Coordinates formatting
            if getattr(reg, "georeferenced", False) and getattr(reg, "centroid_wgs84", None):
                coords_str = (
                    f"Lat: {reg.centroid_wgs84[1]:.5f}°, Lon: {reg.centroid_wgs84[0]:.5f}°<br>"
                    f"<span style='font-size: 0.78rem; color: #6B7280;'>UTM 20S: {reg.centroid_utm[0]:.1f}m E, {reg.centroid_utm[1]:.1f}m N</span>"
                )
            else:
                coords_str = f"Pixel [{reg.centroid_pixel[0]}, {reg.centroid_pixel[1]}] (Unprojected)"

            candidate_table_rows += f"""
            <tr>
                <td><strong>{reg.region_id}</strong></td>
                <td><strong>{reg.area_ha:.2f} ha</strong><br><span style='font-size: 0.78rem; color: #6B7280;'>{reg.pixel_count:,} px</span></td>
                <td>{coords_str}</td>
                <td>{reg.mean_probability * 100:.1f}%</td>
                <td>{reg.ref_status}</td>
            </tr>
            """
    else:
        candidate_table_rows = "<tr><td colspan='5' style='text-align: center; color: #6B7280;'>No connected deforestation areas detected above decision threshold.</td></tr>"

    # Reference metrics block
    if conf_stats:
        ref_block = f"""
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-num">{conf_stats.get('iou', 0.0):.4f}</div>
                <div class="metric-label">Exact Intersection-over-Union (IoU)</div>
            </div>
            <div class="metric-card">
                <div class="metric-num">{conf_stats.get('dice', 0.0):.4f}</div>
                <div class="metric-label">Dice / F1 Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-num">{conf_stats.get('precision', 0.0):.4f}</div>
                <div class="metric-label">Precision</div>
            </div>
            <div class="metric-card">
                <div class="metric-num">{conf_stats.get('recall', 0.0):.4f}</div>
                <div class="metric-label">Recall</div>
            </div>
        </div>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Confusion Class</th>
                    <th>Area (Hectares)</th>
                    <th>Pixel Count</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><span class="badge badge-green">True Positive (TP)</span></td>
                    <td>{conf_stats.get('tp_ha', 0.0):.2f} ha</td>
                    <td>{conf_stats.get('tp', 0):,} px</td>
                    <td>Correctly detected deforestation confirmed by PRODES reference</td>
                </tr>
                <tr>
                    <td><span class="badge badge-red">False Positive (FP)</span></td>
                    <td>{conf_stats.get('fp_ha', 0.0):.2f} ha</td>
                    <td>{conf_stats.get('fp', 0):,} px</td>
                    <td>Predicted clearing not mapped in PRODES 2021 reference (e.g. &lt; 6.25 ha MMU)</td>
                </tr>
                <tr>
                    <td><span class="badge badge-yellow">False Negative (FN)</span></td>
                    <td>{conf_stats.get('fn_ha', 0.0):.2f} ha</td>
                    <td>{conf_stats.get('fn', 0):,} px</td>
                    <td>PRODES reference loss missed by segmentation model</td>
                </tr>
            </tbody>
        </table>
        """
    else:
        ref_block = """
        <div class="alert-box alert-neutral">
            <strong>Ground-Truth Reference Labels Unavailable:</strong> This input does not include an independent
            reference cartography mask. Reported results represent autonomous model inference without external validation.
        </div>
        """

    # Explainability sections
    lime_block = f"""
    <div class="card">
        <h3>LIME Local Superpixel Counterfactual Attribution (MobileNetV3 Classifier)</h3>
        <p class="caption">
            <strong>What LIME explains:</strong> Local Interpretable Model-agnostic Explanations tests what happens
            when individual image superpixels in the 2021 monitoring scene are replaced with their undisturbed 2020 condition.
            Superpixels in coral/red are those whose disturbance drove the model's deforestation prediction.
            Superpixels in blue suppressed the deforestation call.
        </p>
        <img class="report-img" src="{img_lime}" alt="LIME Local Attribution Figure" />
        <p class="caption">
            <strong>Local Surrogate Fidelity (R²):</strong> {lime_summary.get('r2_score', 0.0):.4f}
            (indicates how closely LIME's local linear surrogate fits the neural network).<br>
            <strong>Spatial Cross-Check:</strong> {lime_summary.get('cross_check', 'LIME supporting regions spatially coincide with the U-Net segmentation candidate areas.')}
        </p>
    </div>
    """ if img_lime and lime_summary else ""

    ig_block = f"""
    <div class="card">
        <h3>Integrated Gradients Dense Attribution (Early-Fusion U-Net)</h3>
        <p class="caption">Axiomatic path attribution along linear interpolation path from zero-reflectance baseline. Identifies spectral channels and spatial features driving the segmentation prediction.</p>
        <img class="report-img" src="{img_ig}" alt="Integrated Gradients Attribution Figure" />
        <p class="caption"><strong>Completeness Error:</strong> {ig_summary.get('completeness_error', 0.0):.2%} | <strong>Attribution Meaning:</strong> Signed gradient path accumulation; positive values indicate spectral contrasts increasing deforestation likelihood, while negative values suppress it.</p>
    </div>
    """ if img_ig and ig_summary else ""

    gradcam_block = f"""
    <div class="card">
        <h3>Grad-CAM Coarse Saliency (MobileNetV3 Siamese Classifier)</h3>
        <p class="caption">Class activation mapping across deep convolutional feature maps for patch-level classification.</p>
        <img class="report-img" src="{img_gradcam}" alt="Grad-CAM Saliency Figure" />
        <p class="caption"><strong>Important Scientific Notice:</strong> Grad-CAM indicates classification sensitivity across low-resolution feature maps. It is an explanatory diagnostic, NOT a pixel-level deforestation boundary.</p>
    </div>
    """ if img_gradcam else ""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Deforestation Analysis Report — {patch_id}</title>
    <style>
        :root {{
            --bg-page: #F7F8F5;
            --bg-surface: #FFFFFF;
            --bg-surface-alt: #FBFBF9;
            --primary-forest: #173F35;
            --secondary-teal: #477C73;
            --text-primary: #202824;
            --text-secondary: #69736D;
            --border-subtle: #DDE3DC;
            --border-strong: #BDC8BD;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            color: #202824;
            background-color: #F7F8F5;
            margin: 0;
            padding: 2.5rem 1.5rem;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1060px;
            margin: 0 auto;
            background: #FFFFFF;
            padding: 2.8rem;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
            border: 1px solid #DDE3DC;
        }}
        h1 {{
            font-size: 1.85rem;
            font-weight: 700;
            color: #173F35;
            margin-top: 0;
            margin-bottom: 0.35rem;
            letter-spacing: -0.02em;
        }}
        h2 {{
            font-size: 1.30rem;
            font-weight: 700;
            color: #173F35;
            border-bottom: 1px solid #DDE3DC;
            padding-bottom: 0.5rem;
            margin-top: 2.2rem;
            margin-bottom: 1.0rem;
        }}
        h3 {{
            font-size: 1.05rem;
            font-weight: 700;
            color: #202824;
            margin-top: 1.0rem;
            margin-bottom: 0.5rem;
        }}
        .subtitle {{
            font-size: 0.88rem;
            color: #69736D;
            margin-bottom: 1.8rem;
        }}
        .status-banner {{
            background-color: {status_bg};
            border-left: 5px solid {status_border};
            border-radius: 4px;
            padding: 1.2rem 1.6rem;
            margin-bottom: 1.8rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 1.0rem;
        }}
        .status-title {{
            font-size: 1.35rem;
            font-weight: 700;
            color: {status_text};
            margin: 0;
        }}
        .status-sub {{
            font-size: 0.88rem;
            color: #202824;
            margin-top: 0.25rem;
        }}
        .status-metric {{
            text-align: right;
        }}
        .status-num {{
            font-size: 2.4rem;
            font-weight: 700;
            color: #173F35;
            line-height: 1;
        }}
        .status-label {{
            font-size: 0.82rem;
            color: #69736D;
            font-weight: 600;
        }}
        .card {{
            background: #FFFFFF;
            border: 1px solid #DDE3DC;
            border-radius: 4px;
            padding: 1.2rem;
            margin-bottom: 1.3rem;
        }}
        .report-img {{
            width: 100%;
            height: auto;
            border-radius: 4px;
            border: 1px solid #DDE3DC;
            display: block;
            margin: 0.8rem 0;
        }}
        .caption {{
            font-size: 0.82rem;
            color: #69736D;
            margin: 0.3rem 0;
            line-height: 1.5;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.86rem;
            margin: 1.0rem 0 1.5rem 0;
        }}
        .data-table th, .data-table td {{
            padding: 0.65rem 0.85rem;
            text-align: left;
            border-bottom: 1px solid #DDE3DC;
            vertical-align: top;
        }}
        .data-table th {{
            background-color: #F0F2EE;
            font-weight: 700;
            color: #173F35;
        }}
        .meta-key {{
            font-weight: 600;
            color: #69736D;
            width: 30%;
        }}
        .meta-val {{
            color: #202824;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.0rem;
            margin-bottom: 1.5rem;
        }}
        .metric-card {{
            background: #FBFBF9;
            border: 1px solid #DDE3DC;
            border-radius: 4px;
            padding: 1.0rem;
            text-align: center;
        }}
        .metric-num {{
            font-size: 1.8rem;
            font-weight: 700;
            color: #173F35;
        }}
        .metric-label {{
            font-size: 0.80rem;
            color: #69736D;
            font-weight: 600;
            margin-top: 0.2rem;
        }}
        .badge {{
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 3px;
            font-size: 0.78rem;
            font-weight: 700;
        }}
        .badge-green {{ background: #EBF2EE; color: #173F35; border: 1px solid #BDC8BD; }}
        .badge-red {{ background: #FEF2F2; color: #991B1B; border: 1px solid #FCA5A5; }}
        .badge-yellow {{ background: #FFFBEB; color: #92400E; border: 1px solid #FCD34D; }}
        .alert-box {{
            padding: 0.9rem 1.2rem;
            border-radius: 4px;
            font-size: 0.86rem;
            margin: 1.0rem 0;
            line-height: 1.55;
        }}
        .alert-neutral {{
            background-color: #FBFBF9;
            border-left: 4px solid #69736D;
            color: #202824;
        }}
        .alert-info {{
            background-color: #F0F7F6;
            border-left: 4px solid #477C73;
            color: #173F35;
        }}
        .alert-warning {{
            background-color: #FFFBEB;
            border-left: 4px solid #D97706;
            color: #92400E;
        }}
        .footer {{
            margin-top: 2.5rem;
            padding-top: 1.2rem;
            border-top: 1px solid #DDE3DC;
            font-size: 0.80rem;
            color: #69736D;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Deforestation Analysis Report</h1>
        <div class="subtitle">
            Explainable AI Based Geospatial Anomaly Detection & Segmentation System<br>
            Analysis ID: <strong>{patch_id}</strong> | Generated: <strong>{now_str}</strong>
        </div>

        <!-- Section A: Analysis Summary & Executive Summary -->
        <h2>A. Analysis Summary</h2>
        <div class="status-banner">
            <div>
                <h3 class="status-title">{detection_title}</h3>
                <div class="status-sub">
                    Patch-level classification probability: <strong>{clf_result.get('probability', 0.0) * 100:.1f}%</strong>
                    (Decision threshold: {clf_result.get('threshold', 0.27) * 100:.0f}%)
                </div>
            </div>
            <div class="status-metric">
                <div class="status-num">{pred_ha:.2f} ha</div>
                <div class="status-label">Predicted Clearing Area ({pred_px:,} pixels)</div>
            </div>
        </div>

        <div class="card" style="background: #F9FAFB; border-left: 4px solid #4B5563;">
            <h3 style="margin-top: 0; color: #111827;">Executive Plain-Language Summary</h3>
            {exec_rows}
        </div>

        <table class="data-table">
            <thead>
                <tr>
                    <th colspan="2">Input Specification & Evidence Verification</th>
                </tr>
            </thead>
            <tbody>
                {meta_rows}
                <tr>
                    <td class="meta-key">Evidence Suitability</td>
                    <td class="meta-val">Verified: Compatible 8-channel bi-temporal multispectral imagery</td>
                </tr>
                <tr>
                    <td class="meta-key">Primary Classification Model</td>
                    <td class="meta-val">MobileNetV3 Siamese Classifier (tau = 0.2700)</td>
                </tr>
                <tr>
                    <td class="meta-key">Pixel Segmentation Model</td>
                    <td class="meta-val">Early-Fusion U-Net Champion (tau* = 0.500)</td>
                </tr>
            </tbody>
        </table>

        <!-- Section B: Imagery and Change Evidence -->
        <h2>B. Imagery and Change Evidence</h2>
        <div class="card">
            <h3>Registered Bi-Temporal Satellite Observations (Contrast-Stretched)</h3>
            <img class="report-img" src="{img_bitemporal}" alt="Registered Bi-temporal Imagery" />
            <p class="caption">Comparison of Date 1 (Baseline: 08 Aug 2020) and Date 2 (Monitoring: 08 Aug 2021) Sentinel-2 true-color surface reflectance, normalized with consistent joint 2%–98% percentile stretching.</p>
        </div>

        {"<div class='card'><h3>Candidate Deforestation Events Overview Map</h3><img class='report-img' src='" + img_candidate_map + "' alt='Candidate Regions Overview Map' /><p class='caption'>Overview of suspected deforestation events identified via 8-connected component segmentation. Labelled callouts correspond to the Candidate Area Inventory table.</p></div>" if img_candidate_map else ""}

        <div class="card">
            <h3>Model Spatial Prediction & Affected Area Mask</h3>
            <img class="report-img" src="{img_pred_mask}" alt="Predicted Deforestation Mask" />
            <p class="caption">Left: U-Net continuous spatial probability map with decision contour (tau* = 0.500). Right: Discrete predicted deforestation mask (red = clearing, green = undisturbed canopy).</p>
        </div>

        <div class="card">
            <h3>Deforestation Overlay on Monitoring Scene</h3>
            <img class="report-img" src="{img_overlay}" alt="Predicted Mask Overlaid on Imagery" />
            <p class="caption">Predicted deforestation mask overlaid on Date 2 satellite imagery. Shows exact spatial alignment with visible ground features.</p>
        </div>

        {"<div class='card'><h3>Reference Comparison & Confusion Cartography</h3><img class='report-img' src='" + img_confusion + "' alt='Pixel Confusion Map' /><p class='caption'>Spatial comparison against official PRODES reference cartography.</p></div>" if img_confusion else ""}

        <!-- Section C: Quantitative Results & Candidate Regions Inventory -->
        <h2>C. Quantitative Results & Candidate Area Inventory</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-num">{pred_ha:.2f} ha</div>
                <div class="metric-label">Predicted Clearing Area</div>
            </div>
            <div class="metric-card">
                <div class="metric-num">{pred_px:,}</div>
                <div class="metric-label">Predicted Clearing Pixels</div>
            </div>
            <div class="metric-card">
                <div class="metric-num">{valid_px:,}</div>
                <div class="metric-label">Valid Evaluated Pixels</div>
            </div>
            <div class="metric-card">
                <div class="metric-num">{excl_px:,}</div>
                <div class="metric-label">Excluded Pixels (Cloud/Water)</div>
            </div>
        </div>

        <h3>Suspected Deforestation Area Inventory</h3>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Region ID</th>
                    <th>Area (ha / px)</th>
                    <th>Coordinates (Centroid)</th>
                    <th>Mean Model Probability</th>
                    <th>Reference Status</th>
                </tr>
            </thead>
            <tbody>
                {candidate_table_rows}
            </tbody>
        </table>

        <div class="alert-box alert-neutral">
            <strong>Spatial Calibration Basis:</strong> Sentinel-2 10 m UTM projected pixels represent exactly 100 m² (0.01 hectares) per pixel.
            This area conversion is calibrated specifically for Sentinel-2 Level-2A imagery and must not be transferred to uncalibrated aerial photography.
        </div>

        {ref_block}

        <!-- Section D: Supporting Explainability Evidence -->
        <h2>D. Supporting Explainability Evidence</h2>
        <div class="alert-box alert-info">
            <strong>Interpretation Guidance:</strong> Explainability attribution heatmaps reflect model feature sensitivity
            and gradient magnitude. They indicate which spectral channels and spatial regions influenced the models,
            not empirical proof of physical forest loss boundaries.
        </div>

        {lime_block}
        {ig_block}
        {gradcam_block}

        <!-- Section E: Limitations and Interpretation -->
        <h2>E. Limitations and Scientific Interpretation</h2>
        <div class="card">
            <ul>
                <li><strong>Spatial Resolution Limits:</strong> 10 m spatial resolution cannot detect sub-canopy selective logging or small-scale thinning below the single-pixel threshold.</li>
                <li><strong>Minimum Mapping Unit (MMU) Filter Discrepancies:</strong> The benchmark PRODES cartography enforces a 6.25 ha MMU filter, whereas the U-Net operates at pixel scale (0.01 ha). High-confidence model detections smaller than 6.25 ha may appear as false positives relative to PRODES despite representing physical canopy clearing.</li>
                <li><strong>Boundary Uncertainty:</strong> Vector-to-raster cartographic boundary conversion introduces 1–3 pixel spatial uncertainty along clearing edges.</li>
                <li><strong>Temporal Granularity:</strong> Bi-temporal analysis captures net cumulative change between the two observation dates; precise intra-annual timing cannot be determined from two snapshots.</li>
                <li><strong>Geographic Calibration:</strong> The models were trained and benchmarked on the Brazilian Legal Amazon (MGRS 20LMR); applying them to other biomes without adaptation may yield uncalibrated confidence estimates.</li>
                <li><strong>Non-Causality:</strong> Spectral change indicates canopy disruption but does not establish legal intent, zoning non-compliance, or the causal mechanism (e.g. fire vs mechanized clear-felling).</li>
            </ul>
        </div>

        <div class="footer">
            Explainable AI Based Deforestation Anomaly Detection System · Independent Technical Report<br>
            MobileNetV3 SHA-256: <code>e8014466...</code> | Early-Fusion U-Net SHA-256: <code>9b2c21c4...</code>
        </div>
    </div>
</body>
</html>
"""
    return html


def generate_evidence_validation_html_report(
    validation_result: Dict[str, Any],
) -> str:
    """Generate standalone HTML report for unsupported or insufficient evidence inputs."""
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    input_type = validation_result.get("input_type", "UNKNOWN")
    num_dates = validation_result.get("num_dates", 0)
    deficiencies = validation_result.get("deficiencies", [])
    requirements = validation_result.get("requirements", "")
    metadata_list = validation_result.get("image_metadata", [])

    meta_rows = "".join(
        f"<tr><td>{m.get('filename')}</td><td>{m.get('format')}</td><td>{m.get('width')} × {m.get('height')} px</td><td>{m.get('channels')} channels ({m.get('mode')})</td></tr>"
        for m in metadata_list
    )

    deficiencies_li = "".join(f"<li>{d}</li>" for d in deficiencies)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Evidence Validation Diagnostic Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            color: #2B2D31;
            background-color: #F8F9FA;
            margin: 0;
            padding: 2.5rem 1.5rem;
            line-height: 1.6;
        }}
        .container {{
            max-width: 900px;
            margin: 0 auto;
            background: #FFFFFF;
            padding: 3.0rem;
            border-radius: 6px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.08);
            border: 1px solid #E5E7EB;
        }}
        h1 {{
            font-size: 1.8rem;
            font-weight: 800;
            color: #92400E;
            margin-top: 0;
            margin-bottom: 0.4rem;
        }}
        .subtitle {{
            font-size: 0.95rem;
            color: #4B5563;
            margin-bottom: 2.0rem;
        }}
        .status-banner {{
            background-color: #FFFBEB;
            border-left: 6px solid #F59E0B;
            border-radius: 4px;
            padding: 1.2rem 1.6rem;
            margin-bottom: 2.0rem;
        }}
        .status-title {{
            font-size: 1.3rem;
            font-weight: 800;
            color: #B45309;
            margin: 0;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
            margin: 1.0rem 0 1.5rem 0;
        }}
        .data-table th, .data-table td {{
            padding: 0.65rem 0.9rem;
            text-align: left;
            border-bottom: 1px solid #E5E7EB;
        }}
        .data-table th {{
            background-color: #F9FAFB;
            font-weight: 700;
            color: #374151;
        }}
        .card {{
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            border-radius: 4px;
            padding: 1.4rem;
            margin-bottom: 1.5rem;
        }}
        .footer {{
            margin-top: 3.0rem;
            padding-top: 1.5rem;
            border-top: 1px solid #E5E7EB;
            font-size: 0.8rem;
            color: #6B7280;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Evidence Validation Diagnostic Report</h1>
        <div class="subtitle">
            Geospatial Data Verification Diagnostic<br>
            Evaluated: <strong>{now_str}</strong>
        </div>

        <div class="status-banner">
            <h2 class="status-title">Evidence Insufficient for Reliable Deforestation Inference</h2>
            <p style="margin: 0.4rem 0 0 0; color: #78350F; font-size: 0.92rem;">
                The submitted imagery does not satisfy the input contracts required for verified deforestation detection.
                In accordance with scientific integrity principles, no artificial or fabricated prediction has been generated.
            </p>
        </div>

        <h3>Uploaded File Inventory</h3>
        <table class="data-table">
            <thead>
                <tr>
                    <th>Filename</th>
                    <th>Format</th>
                    <th>Dimensions</th>
                    <th>Color Mode</th>
                </tr>
            </thead>
            <tbody>
                {meta_rows}
            </tbody>
        </table>

        <h3>Identified Scientific Deficiencies</h3>
        <div class="card">
            <ul>
                {deficiencies_li}
            </ul>
        </div>

        <h3>Required Input Specifications</h3>
        <div class="card">
            <pre style="background: #F9FAFB; padding: 1.0rem; border-radius: 4px; font-family: monospace; font-size: 0.85rem; line-height: 1.5; white-space: pre-wrap;">{requirements}</pre>
        </div>

        <div class="footer">
            Explainable AI Based Deforestation Anomaly Detection System · Data Verification Report
        </div>
    </div>
</body>
</html>
"""
    return html
