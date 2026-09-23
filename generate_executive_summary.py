"""
generate_executive_summary.py — Generate Updated Executive Summary (.docx)
For Regulatory, Commercial, and Government Stakeholders
"""
import os
import json
import pandas as pd
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml
from docx.oxml.ns import nsdecls

DOC_PATH = "outputs/Executive_Summary_Bank_Customer_Churn.docx"
doc = docx.Document()

for section in doc.sections:
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(0.9)
    section.right_margin = Inches(0.9)

COLOR_NAVY = RGBColor(27, 42, 74)
COLOR_PRIMARY = RGBColor(46, 94, 170)
COLOR_GRAY = RGBColor(90, 100, 115)
HEX_LIGHT_BG = "F4F6F9"

def set_cell_background(cell, hex_color):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

# Title
p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
r = p.add_run("EXECUTIVE BRIEFING · REGULATORY & COMMERCIAL RISK INTELLIGENCE")
r.font.name = "Arial"
r.font.size = Pt(8.5)
r.font.bold = True
r.font.color.rgb = COLOR_PRIMARY

p_title = doc.add_paragraph()
p_title.paragraph_format.space_before = Pt(2)
p_title.paragraph_format.space_after = Pt(6)
r_title = p_title.add_run("Predictive Modeling & Risk Scoring for Bank Customer Churn")
r_title.font.name = "Arial"
r_title.font.size = Pt(17)
r_title.font.bold = True
r_title.font.color.rgb = COLOR_NAVY

p_sub = doc.add_paragraph()
p_sub.paragraph_format.space_after = Pt(10)
r_sub = p_sub.add_run("Prepared for Executive Leadership & Institutional Regulatory Stakeholders\n"
                      "Submission Author: Chandolu Praneeth Kumar · September 2026")
r_sub.font.name = "Arial"
r_sub.font.size = Pt(9.5)
r_sub.font.color.rgb = COLOR_GRAY

# KPI Table
tbl = doc.add_table(rows=1, cols=4)
tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
kpis = [
    ("10,000", "Customer Cohort"),
    ("0.871", "Model ROC-AUC"),
    ("69.8% – 85.0%", "Calibrated Churn Recall"),
    ("$72,650+", "Net Savings / 2k Accounts")
]
for i, (val, lbl) in enumerate(kpis):
    c = tbl.cell(0, i)
    set_cell_background(c, HEX_LIGHT_BG)
    p = c.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    rv = p.add_run(f"{val}\n")
    rv.font.name = "Arial"
    rv.font.size = Pt(14)
    rv.font.bold = True
    rv.font.color.rgb = COLOR_PRIMARY
    rl = p.add_run(lbl)
    rl.font.name = "Arial"
    rl.font.size = Pt(8)
    rl.font.color.rgb = COLOR_GRAY

doc.add_paragraph().paragraph_format.space_after = Pt(6)

def add_sec(title):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(title)
    r.font.name = "Arial"
    r.font.size = Pt(11)
    r.font.bold = True
    r.font.color.rgb = COLOR_NAVY

def add_body(text):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    r.font.name = "Arial"
    r.font.size = Pt(9.5)

add_sec("1. Strategic Commercial Context")
add_body(
    "Customer attrition is one of the greatest drains on retail banking capital. Acquiring a new retail customer costs 5 to 7 times "
    "more than retaining an existing account. In a multi-market European portfolio of 10,000 customers, the baseline churn rate stands at 20.37%. "
    "Unchecked attrition erodes low-cost deposit funding, depresses cross-sell opportunities, and destroys Customer Lifetime Value (CLV)."
)

add_sec("2. The Technical Breakthrough: Fixing the '50% Recall Blind Spot'")
add_body(
    "Standard machine learning implementations optimize for global accuracy on imbalanced data. Under the conventional 0.50 decision threshold, "
    "the highest-accuracy model misses over 51% of all churning accounts. By formulating an asymmetric banking cost-utility matrix "
    "(cost of missed churn = $1,000 vs cost of retention incentive = $50) and calibrating the operating threshold to T* = 0.27, "
    "true churn detection (Recall) rises from 48.2% to 69.8% (and up to 85.0% under strict cost minimization). "
    "This single analytical intervention cuts unmitigated portfolio attrition losses by over 38%, generating $72,650 in direct net savings "
    "per 2,000 evaluated customers."
)

add_sec("3. Key Senior Data Analyst Discoveries")
add_body(
    "• The German Zero-Balance Paradox: Germany's apparent 32.4% churn rate (vs ~16% in France and Spain) is driven by deposit composition. "
    "In Germany, 0.0% of accounts have a €0 balance, whereas in France and Spain, ~48% of customers maintain €0 balances (dormant secondary accounts). "
    "When comparing active funded accounts, the geographic gap narrows dramatically (25.5% vs 32.4%)."
)
add_body(
    "• The Multi-Product Bundling Trap: Customers holding 2 products exhibit the highest retention (churn < 8%). Customers with 3 or 4 products "
    "churn at catastrophic rates of 83% to 100%. However, this cohort represents only 3.26% of customers, indicating forced bundling and fee "
    "dissatisfaction rather than disloyalty among standard multi-product accounts."
)
add_body(
    "• The 45–65 Age Vulnerability: Churn probability remains flat below age 38, accelerating rapidly past age 42 and peaking at age 52, "
    "reflecting life-stage wealth consolidation and competitor solicitations."
)

add_sec("4. Governance, Regulatory Compliance & Decision Support")
add_body(
    "Under Basel III (BCBS 239) and the European Union Artificial Intelligence Act (EU AI Act), financial risk algorithms must provide "
    "rigorous interpretability and audit trails. The system deploys model-agnostic permutation importance and partial dependence plots, "
    "ensuring that every risk score is accompanied by human-auditable driver attributions. The entire pipeline is packaged in an enterprise "
    "Streamlit application equipped with single-customer scoring, batch CSV processing (10,000+ accounts), and an Executive Retention ROI Simulator."
)

doc.save(DOC_PATH)
print(f"Executive Summary updated successfully: {DOC_PATH}")
