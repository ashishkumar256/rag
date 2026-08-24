"""
Generates 20 synthetic, multi-page PDFs for testing a RAG pipeline.

Each PDF is a fictional "company report" with specific facts embedded
(revenue figures, employee names, dates). Because the facts are known,
you can ask the RAG system questions and check whether it retrieves
the *correct* chunk and gives the *correct* answer -- which is much
harder to judge with real documents where you don't already know the answer.

Run: python generate_test_pdfs.py
Output: ./test_pdfs/company_01.pdf ... company_20.pdf
"""

import random
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch

OUT_DIR = Path(__file__).parent / "test_pdfs"
OUT_DIR.mkdir(exist_ok=True)

random.seed(42)  # reproducible facts across runs

COMPANY_NAMES = [
    "Northwind Robotics", "Bluepeak Logistics", "Cedarline Foods", "Ashgrove Analytics",
    "Ferrowatt Energy", "Millbrook Textiles", "Solace Health Systems", "Ironvale Materials",
    "Quillsong Media", "Harborlight Insurance", "Redcliff Mining", "Verdant AgriTech",
    "Stonebridge Capital", "Palefire Semiconductors", "Wrenhollow Publishing",
    "Copperfield Freight", "Larkspur Biotech", "Granite Peak Construction",
    "Silvermoon Retail", "Emberline Aerospace",
]

FIRST_NAMES = ["Maria", "James", "Aisha", "Liu", "Fatima", "Daniel", "Priya", "Tom", "Elena", "Kwame"]
LAST_NAMES = ["Chen", "Okafor", "Torres", "Kowalski", "Nakamura", "Silva", "Patel", "Andersson"]

styles = getSampleStyleSheet()

def make_report(index: int, company: str) -> dict:
    """Builds one fictional report and returns the facts used, so we
    have a ground-truth answer key for testing queries later."""
    ceo = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    cfo = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    year = random.choice([2023, 2024, 2025])
    q3_revenue = round(random.uniform(2.5, 480.0), 1)
    growth_pct = round(random.uniform(-8.0, 35.0), 1)
    employees = random.randint(80, 12000)
    hq_city = random.choice(["Austin", "Leeds", "Nairobi", "Osaka", "Toronto", "Krakow", "Santiago"])
    founded = random.randint(1978, 2019)

    facts = {
        "file": f"company_{index:02d}.pdf",
        "company": company,
        "ceo": ceo,
        "cfo": cfo,
        "year": year,
        "q3_revenue_millions": q3_revenue,
        "growth_pct": growth_pct,
        "employees": employees,
        "hq_city": hq_city,
        "founded": founded,
    }

    doc = SimpleDocTemplate(str(OUT_DIR / facts["file"]), pagesize=letter,
                             topMargin=0.8*inch, bottomMargin=0.8*inch)
    story = []

    # Page 1: Title + Executive Summary
    story.append(Paragraph(f"{company}", styles["Title"]))
    story.append(Paragraph(f"Annual Report {year}", styles["Heading2"]))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Executive Summary", styles["Heading1"]))
    story.append(Paragraph(
        f"{company} was founded in {founded} and is headquartered in {hq_city}. "
        f"The company is led by Chief Executive Officer {ceo}, with financial "
        f"oversight provided by Chief Financial Officer {cfo}. As of the close of "
        f"fiscal year {year}, {company} employed approximately {employees:,} people "
        f"across its global operations. This report summarizes financial performance, "
        f"operational highlights, and strategic priorities for the year ahead.",
        styles["Normal"]))
    story.append(PageBreak())

    # Page 2: Financial Performance (the key fact-bearing page)
    story.append(Paragraph("Financial Performance", styles["Heading1"]))
    story.append(Paragraph(
        f"In the third quarter of {year}, {company} reported revenue of "
        f"${q3_revenue} million, representing a year-over-year growth rate of "
        f"{growth_pct}% compared to the same quarter in the prior year. "
        f"Management attributes this performance to continued demand in core "
        f"markets and disciplined cost management across business units.",
        styles["Normal"]))
    story.append(Spacer(1, 12))

    table_data = [
        ["Metric", "Value"],
        ["Q3 Revenue", f"${q3_revenue}M"],
        ["YoY Growth", f"{growth_pct}%"],
        ["Employees", f"{employees:,}"],
        ["Headquarters", hq_city],
        ["Founded", str(founded)],
    ]
    t = Table(table_data, colWidths=[2.5*inch, 2.5*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d3748")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(PageBreak())

    # Page 3: Strategic Priorities (padding content, no key facts --
    # tests whether retrieval correctly ignores irrelevant chunks)
    story.append(Paragraph("Strategic Priorities", styles["Heading1"]))
    story.append(Paragraph(
        f"Looking ahead, {company} will focus on three strategic pillars: "
        f"operational efficiency, sustainable growth, and workforce development. "
        f"The leadership team, under {ceo}'s direction, has committed to "
        f"expanding investment in employee training programs and modernizing "
        f"core infrastructure. The board has expressed confidence in the "
        f"company's long-term trajectory and its ability to navigate evolving "
        f"market conditions while maintaining its commitment to quality and "
        f"customer satisfaction across all regions of operation.",
        styles["Normal"]))

    doc.build(story)
    return facts


if __name__ == "__main__":
    all_facts = []
    for i, company in enumerate(COMPANY_NAMES, start=1):
        facts = make_report(i, company)
        all_facts.append(facts)
        print(f"Created {facts['file']}: {company}")

    # Save the answer key so you can check retrieval accuracy
    import json
    with open(OUT_DIR / "_answer_key.json", "w") as f:
        json.dump(all_facts, f, indent=2)

    print(f"\nDone. {len(all_facts)} PDFs written to {OUT_DIR}")
    print("Answer key (ground truth facts) saved to _answer_key.json")
    print("\nTry questions like:")
    ex = all_facts[0]
    ex_company = ex["company"]
    ex_year = ex["year"]
    print(f'  "What was {ex_company}\'s Q3 revenue in {ex_year}?"')
    print(f'  "Who is the CEO of {ex_company}?"')
