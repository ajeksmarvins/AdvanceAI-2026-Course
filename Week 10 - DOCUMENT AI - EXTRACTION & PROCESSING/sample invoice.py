from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Create a sample invoice
doc = SimpleDocTemplate("sample_invoice.pdf", pagesize=letter)
styles = getSampleStyleSheet()
elements = []

# Title
elements.append(Paragraph("INVOICE", styles['Title']))
elements.append(Spacer(1, 12))

# Company info
elements.append(Paragraph("TechCorp Solutions", styles['Heading2']))
elements.append(Paragraph("123 Business Park, Suite 100", styles['Normal']))
elements.append(Paragraph("San Francisco, CA 94105", styles['Normal']))
elements.append(Paragraph("billing@techcorp.com", styles['Normal']))
elements.append(Spacer(1, 12))

# Invoice details
invoice_data = [
    ["Invoice Number:", "INV-2025-00123"],
    ["Invoice Date:", "March 15, 2025"],
    ["Due Date:", "April 14, 2025"],
    ["Payment Terms:", "Net 30"],
]
details_table = Table(invoice_data, colWidths=[150, 250])
details_table.setStyle(TableStyle([
    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('TEXTCOLOR', (0, 0), (0, -1), colors.grey),
    ('TEXTCOLOR', (1, 0), (1, -1), colors.black),
]))
elements.append(details_table)
elements.append(Spacer(1, 12))

# Bill to
elements.append(Paragraph("Bill To:", styles['Heading2']))
elements.append(Paragraph("Acme Corporation", styles['Normal']))
elements.append(Paragraph("456 Industrial Ave", styles['Normal']))
elements.append(Paragraph("Chicago, IL 60601", styles['Normal']))
elements.append(Spacer(1, 12))

# Line items
items_data = [
    ["Description", "Quantity", "Unit Price", "Total"],
    ["Laptop Pro 16-inch", "2", "$1,299.99", "$2,599.98"],
    ["Wireless Mouse", "5", "$29.99", "$149.95"],
    ["USB-C Docking Station", "3", "$199.99", "$599.97"],
    ["Extended Warranty (3yr)", "2", "$249.99", "$499.98"],
]
items_table = Table(items_data, colWidths=[200, 80, 100, 100])
items_table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2C3E50')),
    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
    ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
]))
elements.append(items_table)
elements.append(Spacer(1, 12))

# Totals
totals_data = [
    ["", "Subtotal:", "$3,849.88"],
    ["", "Tax (8.5%):", "$327.24"],
    ["", "TOTAL:", "$4,177.12"],
]
totals_table = Table(totals_data, colWidths=[280, 100, 100])
totals_table.setStyle(TableStyle([
    ('FONTNAME', (1, 0), (1, -1), 'Helvetica-Bold'),
    ('FONTNAME', (2, 2), (2, 2), 'Helvetica-Bold'),
    ('FONTSIZE', (0, 0), (-1, -1), 10),
    ('TEXTCOLOR', (2, 2), (2, 2), colors.black),
]))
elements.append(totals_table)

doc.build(elements)
print("Sample invoice created: sample_invoice.pdf")