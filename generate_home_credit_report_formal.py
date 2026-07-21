"""Build the formal Home Credit model-development report from run artifacts.

The report contains no embedded performance values. Every metric, sample count,
model name, confidence interval, and validation conclusion is read from the
current notebook outputs.
"""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    Image,
    LongTable,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
PDF_PATH = OUTPUT_DIR / "home_credit_model_comparison_report_formal.pdf"
DOWNLOADS_PATH = Path.home() / "Downloads" / PDF_PATH.name

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN_X = 18 * mm
MARGIN_TOP = 19 * mm
MARGIN_BOTTOM = 17 * mm
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN_X

NAVY = colors.HexColor("#17324D")
BLUE = colors.HexColor("#2F6690")
INK = colors.HexColor("#202A35")
MUTED = colors.HexColor("#64717D")
PALE_BLUE = colors.HexColor("#EAF1F7")
PALE_TEAL = colors.HexColor("#E7F2F0")
PALE_ORANGE = colors.HexColor("#F8EEE7")
LINE = colors.HexColor("#CAD4DD")


def register_fonts() -> tuple[str, str]:
    candidates = [
        (
            Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
            Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        ),
        (
            Path("/Library/Fonts/Arial.ttf"),
            Path("/Library/Fonts/Arial Bold.ttf"),
        ),
    ]
    for regular, bold in candidates:
        if regular.exists() and bold.exists():
            pdfmetrics.registerFont(TTFont("ReportRegular", str(regular)))
            pdfmetrics.registerFont(TTFont("ReportBold", str(bold)))
            return "ReportRegular", "ReportBold"
    return "Helvetica", "Helvetica-Bold"


FONT, FONT_BOLD = register_fonts()


def require_artifacts(names: list[str]) -> None:
    missing = [name for name in names if not (OUTPUT_DIR / name).exists()]
    if missing:
        raise FileNotFoundError(
            "Run home_credit_model_comparison.ipynb before building the report. "
            f"Missing artifacts: {missing}"
        )


REQUIRED_ARTIFACTS = [
    "model_metrics.csv",
    "bootstrap_metric_intervals.csv",
    "paired_bootstrap_auc_differences.csv",
    "delong_auc_test.csv",
    "data_split_summary.csv",
    "best_parameters.csv",
    "train_selection_performance_gap.csv",
    "calibration_diagnostics.csv",
    "selected_calibration_methods.csv",
    "segment_performance.csv",
    "random_split_psi.csv",
    "cost_sensitivity.csv",
    "decile_quality_summary.csv",
    "selected_features.csv",
    "dropped_features.csv",
    "validation_checks.csv",
    "warnings_log.csv",
    "run_metadata.json",
]
require_artifacts(REQUIRED_ARTIFACTS)


metrics = pd.read_csv(OUTPUT_DIR / "model_metrics.csv").sort_values("AUC", ascending=False)
bootstrap = pd.read_csv(OUTPUT_DIR / "bootstrap_metric_intervals.csv")
paired_bootstrap = pd.read_csv(OUTPUT_DIR / "paired_bootstrap_auc_differences.csv")
delong = pd.read_csv(OUTPUT_DIR / "delong_auc_test.csv")
splits = pd.read_csv(OUTPUT_DIR / "data_split_summary.csv")
parameters = pd.read_csv(OUTPUT_DIR / "best_parameters.csv")
gaps = pd.read_csv(OUTPUT_DIR / "train_selection_performance_gap.csv")
calibration = pd.read_csv(OUTPUT_DIR / "calibration_diagnostics.csv")
selected_calibration = pd.read_csv(OUTPUT_DIR / "selected_calibration_methods.csv")
segments = pd.read_csv(OUTPUT_DIR / "segment_performance.csv")
psi = pd.read_csv(OUTPUT_DIR / "random_split_psi.csv")
costs = pd.read_csv(OUTPUT_DIR / "cost_sensitivity.csv")
deciles = pd.read_csv(OUTPUT_DIR / "decile_quality_summary.csv")
selected_features = pd.read_csv(OUTPUT_DIR / "selected_features.csv")
dropped_features = pd.read_csv(OUTPUT_DIR / "dropped_features.csv")
checks = pd.read_csv(OUTPUT_DIR / "validation_checks.csv")
warnings_log = pd.read_csv(OUTPUT_DIR / "warnings_log.csv")
metadata = json.loads((OUTPUT_DIR / "run_metadata.json").read_text(encoding="utf-8"))

if metrics["Model"].nunique() != 5:
    raise ValueError("The formal report requires exactly five model results")
if not checks["passed"].astype(bool).all():
    failed = checks.loc[~checks["passed"].astype(bool), "check"].tolist()
    raise ValueError(f"Notebook validation checks failed: {failed}")


styles = getSampleStyleSheet()
styles.add(ParagraphStyle(
    "ReportTitle", fontName=FONT_BOLD, fontSize=25, leading=30,
    textColor=NAVY, alignment=TA_LEFT, spaceAfter=8 * mm,
))
styles.add(ParagraphStyle(
    "ReportSubtitle", fontName=FONT, fontSize=12.5, leading=18,
    textColor=MUTED, alignment=TA_LEFT, spaceAfter=10 * mm,
))
styles.add(ParagraphStyle(
    "H1Custom", fontName=FONT_BOLD, fontSize=17, leading=21,
    textColor=NAVY, spaceBefore=5 * mm, spaceAfter=3.5 * mm,
))
styles.add(ParagraphStyle(
    "H2Custom", fontName=FONT_BOLD, fontSize=12.5, leading=16,
    textColor=BLUE, spaceBefore=4 * mm, spaceAfter=2.2 * mm,
))
styles.add(ParagraphStyle(
    "BodyCustom", fontName=FONT, fontSize=9.4, leading=14,
    textColor=INK, alignment=TA_LEFT, spaceAfter=2.5 * mm,
))
styles.add(ParagraphStyle(
    "SmallCustom", fontName=FONT, fontSize=7.8, leading=10.5,
    textColor=MUTED, spaceAfter=2 * mm,
))
styles.add(ParagraphStyle(
    "CalloutCustom", fontName=FONT, fontSize=9.5, leading=14,
    textColor=INK, leftIndent=4 * mm, rightIndent=4 * mm,
    spaceBefore=2 * mm, spaceAfter=2 * mm,
))
styles.add(ParagraphStyle(
    "TableHeader", fontName=FONT_BOLD, fontSize=7.3, leading=9,
    textColor=colors.white, alignment=TA_CENTER,
))
styles.add(ParagraphStyle(
    "TableCell", fontName=FONT, fontSize=7.1, leading=9,
    textColor=INK, alignment=TA_CENTER,
))
styles.add(ParagraphStyle(
    "TableCellLeft", fontName=FONT, fontSize=7.1, leading=9,
    textColor=INK, alignment=TA_LEFT,
))
styles.add(ParagraphStyle(
    "CaptionCustom", fontName=FONT, fontSize=7.5, leading=10,
    textColor=MUTED, alignment=TA_CENTER, spaceBefore=1.5 * mm,
    spaceAfter=3 * mm,
))


def safe(value: object) -> str:
    return html.escape(str(value))


def body(text: str) -> Paragraph:
    return Paragraph(text, styles["BodyCustom"])


def h1(text: str) -> Paragraph:
    return Paragraph(text, styles["H1Custom"])


def h2(text: str) -> Paragraph:
    return Paragraph(text, styles["H2Custom"])


def bullet(text: str) -> Paragraph:
    return Paragraph(f"<bullet>&bull;</bullet>{text}", styles["BodyCustom"])


class Rule(Flowable):
    def __init__(self, color=LINE, thickness=0.7, space_after=3 * mm):
        super().__init__()
        self.color = color
        self.thickness = thickness
        self.space_after = space_after

    def wrap(self, avail_width, avail_height):
        self.width = avail_width
        return avail_width, self.space_after

    def draw(self):
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, self.space_after - 1, self.width, self.space_after - 1)


def table_from_frame(
    frame: pd.DataFrame,
    columns: list[str],
    labels: list[str] | None = None,
    formats: dict[str, str] | None = None,
    widths: list[float] | None = None,
    left_columns: set[str] | None = None,
) -> LongTable:
    formats = formats or {}
    labels = labels or columns
    left_columns = left_columns or set()
    rows = [[Paragraph(safe(label), styles["TableHeader"]) for label in labels]]
    for _, record in frame[columns].iterrows():
        cells = []
        for column in columns:
            value = record[column]
            if pd.isna(value):
                rendered = ""
            elif column in formats:
                rendered = formats[column].format(value)
            else:
                rendered = str(value)
            style = styles["TableCellLeft"] if column in left_columns else styles["TableCell"]
            cells.append(Paragraph(safe(rendered), style))
        rows.append(cells)
    table = LongTable(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, LINE),
        ("BACKGROUND", (0, 1), (-1, -1), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F8FA")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def image_flowable(path: Path, max_width=CONTENT_WIDTH, max_height=105 * mm) -> Image | None:
    if not path.exists():
        return None
    with PILImage.open(path) as image:
        width, height = image.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def add_figure(story: list, filename: str, caption: str, max_height=105 * mm) -> None:
    figure = image_flowable(FIGURE_DIR / filename, max_height=max_height)
    if figure is None:
        raise FileNotFoundError(f"Required report figure is missing: {FIGURE_DIR / filename}")
    story.extend([
        Spacer(1, 1.5 * mm),
        figure,
        Paragraph(caption, styles["CaptionCustom"]),
    ])


def callout(text: str, color=PALE_BLUE) -> Table:
    content = Paragraph(text, styles["CalloutCustom"])
    table = Table([[content]], colWidths=[CONTENT_WIDTH])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), color),
        ("BOX", (0, 0), (-1, -1), 0.8, BLUE),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return table


def page_header_footer(canvas, document) -> None:
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN_X, PAGE_HEIGHT - 12 * mm, PAGE_WIDTH - MARGIN_X, PAGE_HEIGHT - 12 * mm)
    canvas.setFont(FONT, 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN_X, PAGE_HEIGHT - 9.5 * mm, "HOME CREDIT DEFAULT RISK")
    canvas.drawRightString(PAGE_WIDTH - MARGIN_X, PAGE_HEIGHT - 9.5 * mm, "MODEL DEVELOPMENT REPORT")
    canvas.line(MARGIN_X, 10 * mm, PAGE_WIDTH - MARGIN_X, 10 * mm)
    canvas.drawString(MARGIN_X, 6.8 * mm, "Public-data benchmark | Not a production PD approval")
    canvas.drawRightString(PAGE_WIDTH - MARGIN_X, 6.8 * mm, f"Page {document.page}")
    canvas.restoreState()


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename: Path):
        super().__init__(
            str(filename),
            pagesize=A4,
            leftMargin=MARGIN_X,
            rightMargin=MARGIN_X,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
            title="Home Credit Default Risk: Model Development and Comparison",
            author="Model Development Project",
            subject="Credit risk model comparison with statistical and calibration evidence",
        )
        frame = Frame(
            MARGIN_X,
            MARGIN_BOTTOM,
            CONTENT_WIDTH,
            PAGE_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM,
            id="content",
        )
        self.addPageTemplates(PageTemplate(id="main", frames=[frame], onPage=page_header_footer))


def metric_row(model: str) -> pd.Series:
    return metrics.loc[metrics["Model"] == model].iloc[0]


def auc_interval(model: str) -> pd.Series:
    return bootstrap.loc[(bootstrap["Model"] == model) & (bootstrap["Metric"] == "AUC")].iloc[0]


top_model = str(metrics.iloc[0]["Model"])
runner_up = str(metrics.iloc[1]["Model"])
top_metric = metric_row(top_model)
runner_metric = metric_row(runner_up)
top_ci = auc_interval(top_model)
runner_ci = auc_interval(runner_up)
delong_row = delong.iloc[0]
delong_significant = float(delong_row["PValueTwoSided"]) < 0.05
ci_overlap = max(top_ci["Lower"], runner_ci["Lower"]) <= min(top_ci["Upper"], runner_ci["Upper"])
max_psi = float(psi["PSI"].max())
all_checks_passed = bool(checks["passed"].astype(bool).all())
full_or_debug = "debug" if metadata.get("debug_mode") else "full"


def recommendation_text() -> str:
    if delong_significant:
        inference = "The pre-specified DeLong comparison is statistically significant."
    else:
        inference = "The pre-specified DeLong comparison is not statistically significant at 5%."
    overlap = "overlap" if ci_overlap else "do not overlap"
    return (
        f"<b>Development recommendation:</b> retain {safe(top_model)} as the observed discrimination leader "
        f"and {safe(runner_up)} as the principal challenger. The two leading final-test AUC intervals {overlap}. "
        f"{inference} No production champion is approved from this experiment alone."
    )


story: list = []

# Title and decision page
story.extend([
    Spacer(1, 20 * mm),
    Paragraph("Home Credit Default Risk", styles["ReportTitle"]),
    Paragraph(
        "Model Development, Statistical Comparison, Probability Calibration, and Risk Diagnostics",
        styles["ReportSubtitle"],
    ),
    Rule(color=BLUE, thickness=2, space_after=8 * mm),
    body(
        f"This report is generated from the current <b>{full_or_debug} notebook run</b>. "
        f"It evaluates {len(metrics)} classification models on {int(metadata['source_rows']):,} public application records "
        f"using mutually exclusive development partitions."
    ),
    Spacer(1, 4 * mm),
    callout(recommendation_text(), PALE_TEAL),
    Spacer(1, 7 * mm),
    table_from_frame(
        metrics.head(5),
        ["Model", "AUC", "KS", "PR-AUC", "F1", "Brier", "TrainTime"],
        ["Model", "AUC", "KS", "PR-AUC", "F1", "Raw Brier", "Final fit (s)"],
        {"AUC": "{:.4f}", "KS": "{:.4f}", "PR-AUC": "{:.4f}", "F1": "{:.4f}", "Brier": "{:.4f}", "TrainTime": "{:.2f}"},
        [38 * mm, 18 * mm, 18 * mm, 20 * mm, 18 * mm, 22 * mm, 23 * mm],
        {"Model"},
    ),
    Spacer(1, 6 * mm),
    Paragraph("Report status", styles["H2Custom"]),
    bullet(f"Notebook integrity checks: <b>{'all passed' if all_checks_passed else 'failed'}</b>."),
    bullet(f"Captured fit warnings requiring action: <b>{len(warnings_log)}</b>."),
    bullet("Use classification: model-development evidence on a public benchmark dataset."),
    bullet("Excluded claims: out-of-time stability, reject inference, long-run PD calibration, expected loss, profitability, and regulatory capital adequacy."),
    Spacer(1, 8 * mm),
    Paragraph("Prepared from reproducible output artifacts", styles["SmallCustom"]),
    Paragraph("Evidence directory: outputs/ (relative to the repository root)", styles["SmallCustom"]),
    PageBreak(),
])

# Executive summary
story.extend([
    h1("1. Executive Summary"),
    body(
        f"The highest observed final-test AUC is {top_metric['AUC']:.4f} for <b>{safe(top_model)}</b>, "
        f"with a stratified bootstrap 95% interval of {top_ci['Lower']:.4f} to {top_ci['Upper']:.4f}. "
        f"<b>{safe(runner_up)}</b> follows at {runner_metric['AUC']:.4f}, with an interval of "
        f"{runner_ci['Lower']:.4f} to {runner_ci['Upper']:.4f}. These are observed random-holdout results, "
        "not evidence that either algorithm will dominate across future booking vintages."
    ),
    body(
        f"The pre-specified paired DeLong comparison uses <b>{safe(delong_row['ModelA'])}</b> and "
        f"<b>{safe(delong_row['ModelB'])}</b>, the two candidates chosen by model-selection AUC. "
        f"The AUC difference is {delong_row['AUCDifference']:.4f} with p={delong_row['PValueTwoSided']:.4f}. "
        f"It is {'statistically distinguishable' if delong_significant else 'not statistically distinguishable'} "
        "at the 5% level under this conditional test."
    ),
    body(
        "Class weighting changes the effective event prior used during training. Raw model outputs are therefore "
        "treated as ranking scores rather than accepted portfolio PDs. Prior correction, Platt scaling, and Isotonic "
        "regression are assessed on an independent calibration partition, with calibration-in-the-large, slope, "
        "observed-to-expected ratio, Brier score, and Log Loss reported on the final test."
    ),
    body(
        "A model selection decision should balance discrimination, calibration, stability, explainability, operating "
        "cost, implementation effort, and governance. This report names development candidates only; a production "
        "champion still requires out-of-time testing and portfolio-specific economics."
    ),
    Spacer(1, 1 * mm),
])
add_figure(story, "comparison_roc_curves.png", "Figure 1. Final-test ROC curves for all five models.", 92 * mm)
story.append(PageBreak())

# Data and design
story.extend([
    h1("2. Data, Target, and Experimental Design"),
    h2("2.1 Data scope"),
    body(
        f"The source is Kaggle's Home Credit Default Risk application table. The current run reads "
        f"{int(metadata['source_rows']):,} rows and {int(metadata['source_columns']):,} source columns. "
        "TARGET=1 denotes repayment difficulty in the competition label; it is not a legal bankruptcy, charge-off, "
        "or institution-specific Basel default definition."
    ),
    h2("2.2 Independent data roles"),
    body(
        "The design separates model fitting, boosted-model selection, probability calibration, operating-threshold "
        "selection, and final evaluation. Feature filtering and preprocessing are fitted on training data only."
    ),
    table_from_frame(
        splits,
        list(splits.columns),
        [column.replace("_", " ").title() for column in splits.columns],
        {column: "{:.4f}" for column in splits.columns if "rate" in column.lower()},
        None,
        {splits.columns[0]},
    ),
    Spacer(1, 3 * mm),
    callout(
        "<b>Temporal limitation:</b> the public application table does not provide a reliable booking timestamp for "
        "the target population. The final partition is a stratified random holdout, not an out-of-time sample.",
        PALE_ORANGE,
    ),
])
add_figure(story, "eda_target_rate.png", "Figure 2. Observed target rate in the development data.", 72 * mm)
add_figure(story, "eda_top30_missing_rate.png", "Figure 3. Highest source-feature missing rates.", 85 * mm)
story.append(PageBreak())

# Feature controls
story.extend([
    h1("3. Data Preparation and Feature Controls"),
    body(
        "The notebook replaces known sentinel day values, standardizes non-finite values, retains missingness for "
        "pipeline imputation, and constructs a small set of business-ratio variables. Ratios use protected division "
        "and do not silently preserve infinite values."
    ),
    body(
        f"Training-only screening retains {int(metadata['selected_raw_features']):,} raw and engineered columns and "
        f"produces {int(metadata['encoded_features']):,} encoded model inputs. The process removes near-constant, "
        "extremely incomplete, and highly correlated columns using training statistics only."
    ),
    h2("3.1 Selected feature sample"),
    table_from_frame(
        selected_features.head(20),
        list(selected_features.columns)[:4],
        [column.replace("_", " ").title() for column in list(selected_features.columns)[:4]],
        widths=[44 * mm] + [37 * mm] * (min(4, selected_features.shape[1]) - 1),
        left_columns={selected_features.columns[0]},
    ),
    h2("3.2 Exclusion evidence"),
    body(
        f"The audit table records {len(dropped_features):,} excluded columns and their reasons. TARGET and SK_ID_CURR "
        "are explicitly excluded from model inputs. The complete lists remain available as CSV evidence rather than "
        "being summarized only in narrative text."
    ),
])
add_figure(story, f"{top_model.lower().replace(' ', '_')}_feature_importance.png", f"Figure 4. Built-in feature importance for {top_model}.", 92 * mm)
story.append(PageBreak())

# Model development
story.extend([
    h1("4. Model Development and Parameter Screening"),
    body(
        "Five model families are compared: Logistic Regression, Decision Tree, Random Forest, XGBoost, and LightGBM. "
        "The search is deliberately described as limited parameter screening. Candidate counts and timing are read "
        "from the run, rather than inferred from the size of a nominal parameter space."
    ),
    table_from_frame(
        parameters,
        ["Model", "CandidateCount", "BestSearchAUC", "TuneTime", "TrainTime", "EarlyStoppingBoundaryReached"],
        ["Model", "Candidates", "Search AUC", "Screening (s)", "Final fit (s)", "Boundary reached"],
        {"CandidateCount": "{:.0f}", "BestSearchAUC": "{:.4f}", "TuneTime": "{:.2f}", "TrainTime": "{:.2f}"},
        [37 * mm, 19 * mm, 23 * mm, 25 * mm, 22 * mm, 29 * mm],
        {"Model"},
    ),
    h2("4.1 Comparable timing definition"),
    body(
        "Every reported final training time measures one clean fit on the complete training partition after parameter "
        "selection. XGBoost and LightGBM use early stopping only during candidate screening, then refit the selected "
        "iteration count on all training rows. This aligns the timing definition with the scikit-learn models."
    ),
    h2("4.2 LightGBM row sampling"),
    body(
        "LightGBM sets subsample_freq=1, so candidate values below subsample=1 activate row bagging. The notebook "
        "checks this parameter in the saved final model evidence."
    ),
])
add_figure(story, "comparison_traintime.png", "Figure 5. Clean final-fit time by model.", 80 * mm)
story.append(PageBreak())

# Performance and uncertainty
story.extend([
    h1("5. Final-Test Performance and Uncertainty"),
    body(
        "AUC, Accuracy Ratio, KS, and PR-AUC assess ranking. F1 and related threshold metrics use the maximum-F1 "
        "cutoff selected only on the separate threshold partition. Brier score and Log Loss shown here refer to raw "
        "model outputs and should be interpreted alongside the calibration section."
    ),
    table_from_frame(
        metrics,
        ["Model", "AUC", "AR", "KS", "PR-AUC", "Precision", "Recall", "F1", "Brier", "LogLoss"],
        ["Model", "AUC", "AR", "KS", "PR-AUC", "Precision", "Recall", "F1", "Brier", "Log Loss"],
        {column: "{:.4f}" for column in ["AUC", "AR", "KS", "PR-AUC", "Precision", "Recall", "F1", "Brier", "LogLoss"]},
        [35 * mm] + [15.5 * mm] * 9,
        {"Model"},
    ),
    h2("5.1 Bootstrap confidence intervals"),
    table_from_frame(
        bootstrap.loc[bootstrap["Metric"] == "AUC"].sort_values("Estimate", ascending=False),
        ["Model", "Estimate", "Lower", "Upper", "BootstrapSE", "BootstrapReplicates"],
        ["Model", "AUC", "Lower 95%", "Upper 95%", "Bootstrap SE", "Replicates"],
        {"Estimate": "{:.4f}", "Lower": "{:.4f}", "Upper": "{:.4f}", "BootstrapSE": "{:.4f}", "BootstrapReplicates": "{:.0f}"},
        [42 * mm, 22 * mm, 24 * mm, 24 * mm, 25 * mm, 23 * mm],
        {"Model"},
    ),
])
add_figure(story, "bootstrap_auc_intervals.png", "Figure 6. Stratified bootstrap AUC intervals on the final random holdout.", 88 * mm)
story.append(PageBreak())

# Significance and gaps
story.extend([
    h1("6. Paired Comparison and Generalization Diagnostics"),
    h2("6.1 Pre-specified DeLong comparison"),
    table_from_frame(
        delong,
        ["ModelA", "ModelB", "AUCA", "AUCB", "AUCDifference", "Lower95", "Upper95", "PValueTwoSided"],
        ["Model A", "Model B", "AUC A", "AUC B", "Difference", "Lower 95%", "Upper 95%", "Two-sided p"],
        {column: "{:.4f}" for column in ["AUCA", "AUCB", "AUCDifference", "Lower95", "Upper95", "PValueTwoSided"]},
        [30 * mm, 30 * mm, 17 * mm, 17 * mm, 21 * mm, 21 * mm, 21 * mm, 22 * mm],
        {"ModelA", "ModelB"},
    ),
    body(
        "The pair is determined by model-selection AUC before inspecting final-test ranking. Paired bootstrap results "
        "for every model pair are retained in paired_bootstrap_auc_differences.csv. A p-value does not measure "
        "business materiality or future-vintage stability."
    ),
    h2("6.2 Train-to-model-selection gaps"),
    table_from_frame(
        gaps.sort_values("TrainSelectionGap", ascending=False),
        ["Model", "TrainAUC", "ModelSelectionAUC", "TestAUC", "TrainSelectionGap"],
        ["Model", "Train AUC", "Selection AUC", "Test AUC", "Train-selection gap"],
        {column: "{:.4f}" for column in ["TrainAUC", "ModelSelectionAUC", "TestAUC", "TrainSelectionGap"]},
        [43 * mm, 27 * mm, 30 * mm, 27 * mm, 34 * mm],
        {"Model"},
    ),
    callout(
        "These gaps are continuous diagnostics. The report does not use an arbitrary fixed cutoff to label a model "
        "as overfitted. Ranking stability still requires repeated development samples and out-of-time evidence.",
        PALE_BLUE,
    ),
])
add_figure(story, "comparison_pr_curves.png", "Figure 7. Precision-recall curves under class imbalance.", 88 * mm)
story.append(PageBreak())

# Deciles
top_slug = top_model.lower().replace(" ", "_")
story.extend([
    h1("7. Risk Ranking and Decile Diagnostics"),
    body(
        "Risk deciles test whether higher predicted scores concentrate observed bad outcomes. They are descriptive "
        "on the final random holdout and should not be interpreted as stable production grades."
    ),
    table_from_frame(
        deciles.sort_values("TopDecileBadCapture", ascending=False),
        ["Model", "AdjacentMonotonicIncreaseRate", "TopDecileBadCapture", "TopDecileLift"],
        ["Model", "Adjacent increase rate", "Top-decile bad capture", "Top-decile lift"],
        {"AdjacentMonotonicIncreaseRate": "{:.3f}", "TopDecileBadCapture": "{:.3%}", "TopDecileLift": "{:.2f}"},
        [48 * mm, 39 * mm, 42 * mm, 32 * mm],
        {"Model"},
    ),
])
add_figure(story, f"{top_slug}_decile_bad_rate.png", f"Figure 8. {top_model} observed bad rate by risk decile.", 77 * mm)
add_figure(story, f"{top_slug}_decile_lift.png", f"Figure 9. {top_model} lift by risk decile.", 77 * mm)
story.append(PageBreak())

# Calibration
story.extend([
    h1("8. Probability Calibration and PD Interpretation"),
    body(
        "Balanced class weights and scale_pos_weight alter the effective event prevalence seen by the learner. "
        "Consequently, raw predict_proba output is not assumed to represent the unadjusted portfolio PD. The project "
        "compares analytical prior correction, Platt scaling, and Isotonic regression on a separate calibration sample."
    ),
    table_from_frame(
        selected_calibration,
        list(selected_calibration.columns),
        [column.replace("_", " ").title() for column in selected_calibration.columns],
        {column: "{:.4f}" for column in selected_calibration.columns if selected_calibration[column].dtype.kind in "fc"},
        None,
        {selected_calibration.columns[0]},
    ),
    h2("8.1 Final-test calibration diagnostics"),
    table_from_frame(
        calibration,
        ["Model", "Calibration", "ObservedBadRate", "MeanPredictedPD", "CalibrationInTheLarge", "CalibrationSlope", "ObservedExpectedRatio", "Brier"],
        ["Model", "Method", "Observed", "Mean PD", "CITL", "Slope", "O/E", "Brier"],
        {column: "{:.4f}" for column in ["ObservedBadRate", "MeanPredictedPD", "CalibrationInTheLarge", "CalibrationSlope", "ObservedExpectedRatio", "Brier"]},
        [30 * mm, 24 * mm, 20 * mm, 20 * mm, 19 * mm, 19 * mm, 17 * mm, 18 * mm],
        {"Model", "Calibration"},
    ),
    callout(
        "<b>PD boundary:</b> the sample bad rate is a point-in-time public-data rate, not a through-the-cycle long-run "
        "default rate. No long-run portfolio anchor is available for intercept anchoring, grade-level calibration, or "
        "regulatory PD estimation.",
        PALE_ORANGE,
    ),
])
for calibrated_model in selected_calibration["Model"].astype(str):
    slug = calibrated_model.lower().replace(" ", "_")
    add_figure(story, f"{slug}_calibration_methods.png", f"Calibration comparison for {calibrated_model}.", 75 * mm)
story.append(PageBreak())

# Segments and PSI
reported_segments = segments.loc[segments["AUCReported"].astype(bool)].copy()
segment_summary = (
    reported_segments.groupby(["SegmentVariable", "Model"], as_index=False)
    .agg(MinAUC=("AUC", "min"), MaxAUC=("AUC", "max"), SegmentCount=("SegmentValue", "nunique"))
)
story.extend([
    h1("9. Segment Performance and Random-Split PSI"),
    body(
        "Segment diagnostics cover gender, contract type, age band, and income band where event counts are sufficient. "
        "Ranges flag heterogeneity for review; they do not prove discrimination or unfairness because composition, "
        "sample size, and confidence intervals differ by segment."
    ),
    table_from_frame(
        segment_summary,
        ["SegmentVariable", "Model", "MinAUC", "MaxAUC", "SegmentCount"],
        ["Segment variable", "Model", "Minimum AUC", "Maximum AUC", "Segments"],
        {"MinAUC": "{:.4f}", "MaxAUC": "{:.4f}", "SegmentCount": "{:.0f}"},
        [34 * mm, 39 * mm, 29 * mm, 29 * mm, 24 * mm],
        {"SegmentVariable", "Model"},
    ),
])
add_figure(story, "segment_auc_leading_models.png", "Figure 10. Segment AUC for leading development candidates.", 90 * mm)
story.extend([
    body(
        f"The maximum random-test-versus-training PSI is {max_psi:.4f}. This is a pipeline consistency diagnostic "
        "under a random split. It must not be presented as evidence of temporal population stability."
    ),
])
add_figure(story, "random_split_psi.png", "Figure 11. Highest random-split PSI values.", 82 * mm)
story.append(PageBreak())

# Cost and model choice
story.extend([
    h1("10. Illustrative Decision Costs"),
    body(
        "Threshold selection depends on the relative cost of approving a bad applicant and rejecting a good applicant. "
        "The table below uses stated cost ratios only. It does not estimate expected credit loss, revenue, funding "
        "cost, operating expense, customer lifetime value, or capital consumption."
    ),
    table_from_frame(
        costs,
        ["Model", "CostRatioBadToGood", "SelectedThreshold", "RejectRate", "BadCaptureRate", "IllustrativeCostPerApplicant"],
        ["Model", "Bad:good cost", "Threshold", "Reject rate", "Bad capture", "Cost / applicant"],
        {"CostRatioBadToGood": "{:.1f}", "SelectedThreshold": "{:.4f}", "RejectRate": "{:.2%}", "BadCaptureRate": "{:.2%}", "IllustrativeCostPerApplicant": "{:.4f}"},
        [39 * mm, 24 * mm, 25 * mm, 24 * mm, 25 * mm, 28 * mm],
        {"Model"},
    ),
    h1("11. Model Choice: What Can Be Defended"),
    callout(recommendation_text(), PALE_TEAL),
    h2(f"Why retain {top_model}"),
    bullet(f"It has the highest observed final-test AUC ({top_metric['AUC']:.4f}) and KS ({top_metric['KS']:.4f}) in this run."),
    bullet("Its ranking evidence is supported by bootstrap intervals, paired comparisons, decile diagnostics, and segment tables rather than a single point estimate."),
    bullet("Its final fit and inference artifacts reproduce saved predictions under the notebook checks."),
    h2("Why not declare the alternatives inferior"),
    bullet(f"{runner_up}: observed AUC is close ({runner_metric['AUC']:.4f}); it remains a credible challenger where runtime, implementation, or calibration is preferable."),
    bullet("Logistic Regression: weaker nonlinear capacity, but the strongest transparency benchmark and often easier to govern, explain, and monitor."),
    bullet("Random Forest: can fit richer interactions, but a larger train-selection gap and heavier explanation burden may not be justified by its holdout gain."),
    bullet("Decision Tree: highly readable rules, but the single-tree structure has lower discrimination, coarser probability steps, and higher partition instability."),
    body(
        "The final choice should therefore be framed as a conditional development recommendation. A lender may "
        "reasonably choose a slightly lower-AUC model when explainability, monitoring simplicity, latency, or model-risk "
        "policy outweighs a small and uncertain discrimination difference."
    ),
    PageBreak(),
])

# Limitations and audit trail
story.extend([
    h1("12. Limitations and Required Next Evidence"),
    bullet("Out-of-time validation: unavailable because the application table lacks a reliable target-population booking timeline."),
    bullet("Repeated development samples: not yet run; one split cannot establish ranking stability across random seeds."),
    bullet("Reject inference: not performed because rejected-applicant outcomes and the acceptance mechanism are unavailable."),
    bullet("Portfolio PD anchoring: no long-run default rate supports TTC calibration or intercept anchoring."),
    bullet("Economic and capital analysis: LGD, EAD, margin, cost, funding, capacity, and capital inputs are absent."),
    bullet("Historical-table aggregation: bureau, previous-application, and repayment tables are not implemented and are not presented as completed functionality."),
    bullet("Interpretability: coefficients and built-in feature importance are descriptive; SHAP analysis is not claimed or silently substituted."),
    h1("13. Reproducibility and Audit Trail"),
    body(
        "The notebook validates disjoint identities, finite metrics, prediction-file AUC reproduction, saved-model reload "
        "agreement, feature leakage controls, active LightGBM bagging, required evidence files, model artifacts, and "
        "coverage of bootstrap and segment results."
    ),
    table_from_frame(
        checks,
        ["check", "passed"],
        ["Validation check", "Passed"],
        widths=[135 * mm, 25 * mm],
        left_columns={"check"},
    ),
    h2("13.1 Warning policy"),
    body(
        f"Warnings are captured around fitting and saved to warnings_log.csv; they are not globally hidden. "
        f"The current run records {len(warnings_log)} warning(s). Parameters reported by a library as ignored or "
        "ineffective, and convergence warnings, fail the notebook validation check."
    ),
    h2("13.2 Evidence files"),
    body(
        "Core evidence includes model_metrics.csv, test_predictions.csv, best_parameters.csv, candidate-level boosting "
        "results, bootstrap_metric_intervals.csv, paired_bootstrap_auc_differences.csv, delong_auc_test.csv, "
        "calibration_diagnostics.csv, segment_performance.csv, random_split_psi.csv, cost_sensitivity.csv, saved models, "
        "and this dynamically generated report."
    ),
    PageBreak(),
    h1("14. Conclusion and Production Approval Gates"),
    callout(recommendation_text(), PALE_TEAL),
    Spacer(1, 3 * mm),
    body(
        f"{safe(top_model)} is the observed leader in this run, but the evidence does not support a universal or "
        "production-ready superiority claim. The defensible outcome is a governed challenger set, with final approval "
        "deferred until temporal, stability, calibration, explainability, and economic evidence is available."
    ),
    table_from_frame(
        pd.DataFrame([
            {"Gate": "Temporal stability", "Current status": "Open", "Evidence required": "Out-of-time sample across representative booking vintages"},
            {"Gate": "Ranking stability", "Current status": "Open", "Evidence required": "Repeated development splits and confidence on rank changes"},
            {"Gate": "Portfolio PD calibration", "Current status": "Development only", "Evidence required": "Current portfolio rate and long-run default anchor"},
            {"Gate": "Segment review", "Current status": "Preliminary", "Evidence required": "Larger samples, uncertainty, policy and fairness review"},
            {"Gate": "Decision economics", "Current status": "Illustrative only", "Evidence required": "LGD, EAD, margin, costs, capacity, and capital"},
            {"Gate": "Explainability", "Current status": "Partial", "Evidence required": "Reason-code validation and adverse-action review"},
            {"Gate": "Independent validation", "Current status": "Open", "Evidence required": "Reperformance, challenge, and governance approval"},
        ]),
        ["Gate", "Current status", "Evidence required"],
        ["Approval gate", "Current status", "Evidence required before production"],
        widths=[42 * mm, 33 * mm, 86 * mm],
        left_columns={"Gate", "Current status", "Evidence required"},
    ),
    Spacer(1, 5 * mm),
    Rule(color=BLUE, thickness=1.2, space_after=4 * mm),
    body(
        "This conclusion is intentionally narrower than a formal institutional model validation opinion. It states "
        "what the current experiment supports, what remains unresolved, and which evidence would change the decision."
    ),
])


document = ReportDocTemplate(PDF_PATH)
document.build(story)

DOWNLOADS_PATH.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(PDF_PATH, DOWNLOADS_PATH)

print(f"Formal report generated: {PDF_PATH}")
print(f"Downloads copy updated: {DOWNLOADS_PATH}")
print(f"Observed AUC leader: {top_model} ({top_metric['AUC']:.4f})")
print(f"All notebook checks passed: {all_checks_passed}")
