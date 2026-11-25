# -*- coding: utf-8 -*-
"""把 研究路线.md 转成排版好的 DOCX。只支持这篇文章用到的 Markdown 子集。"""
import os
import re

from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.section import WD_SECTION
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.enum.text import WD_COLOR_INDEX

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "研究路线.md")
OUT = os.path.join(ROOT, "研究路线.docx")

FONT = "Microsoft YaHei"
MONO = "Consolas"
INK = RGBColor(0x1E, 0x22, 0x27)
MUTED = RGBColor(0x6B, 0x70, 0x76)
ACCENT = RGBColor(0xAD, 0x43, 0x1F)
LINK = RGBColor(0x2F, 0x5C, 0x8A)

LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
CODE_RE = re.compile(r"`([^`]+)`")


def set_run_font(run, name=FONT, size=None, color=None, bold=None, italic=None):
    run.font.name = name
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn("w:rFonts"))
    if rFonts is None:
        rFonts = OxmlElement("w:rFonts")
        rPr.append(rFonts)
    rFonts.set(qn("w:ascii"), name)
    rFonts.set(qn("w:hAnsi"), name)
    rFonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)
    if color is not None:
        run.font.color.rgb = color
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def add_hyperlink(paragraph, url, text, size=10.5):
    part = paragraph.part
    r_id = part.relate_to(
        url,
        "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink",
        is_external=True,
    )
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)
    new_run = OxmlElement("w:r")
    rPr = OxmlElement("w:rPr")
    rFonts = OxmlElement("w:rFonts")
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia"):
        rFonts.set(qn(attr), FONT)
    rPr.append(rFonts)
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "2F5C8A")
    rPr.append(color)
    u = OxmlElement("w:u")
    u.set(qn("w:val"), "single")
    rPr.append(u)
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), str(int(size * 2)))
    rPr.append(sz)
    new_run.append(rPr)
    t = OxmlElement("w:t")
    t.text = text
    t.set(qn("xml:space"), "preserve")
    new_run.append(t)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)


def add_inline(paragraph, text, size=10.5, color=INK, bold=False):
    """解析行内链接与反引号代码。"""
    tokens = []
    i = 0
    for m in re.finditer(r"\[([^\]]+)\]\(([^)]+)\)|`([^`]+)`", text):
        if m.start() > i:
            tokens.append(("text", text[i:m.start()]))
        if m.group(1) is not None:
            tokens.append(("link", m.group(1), m.group(2)))
        else:
            tokens.append(("code", m.group(3)))
        i = m.end()
    if i < len(text):
        tokens.append(("text", text[i:]))

    for tok in tokens:
        if tok[0] == "text":
            r = paragraph.add_run(tok[1])
            set_run_font(r, size=size, color=color, bold=bold)
        elif tok[0] == "link":
            add_hyperlink(paragraph, tok[2], tok[1], size=size)
        else:
            r = paragraph.add_run(tok[1])
            set_run_font(r, name=MONO, size=size - 0.8, color=ACCENT)


def add_bottom_border(paragraph, color="D8D3C8", size="6"):
    pPr = paragraph._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), size)
    bottom.set(qn("w:space"), "6")
    bottom.set(qn("w:color"), color)
    pBdr.append(bottom)
    pPr.append(pBdr)


def add_page_number(paragraph):
    run = paragraph.add_run()
    fldChar = OxmlElement("w:fldChar")
    fldChar.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = "PAGE"
    fldEnd = OxmlElement("w:fldChar")
    fldEnd.set(qn("w:fldCharType"), "end")
    run._r.append(fldChar)
    run._r.append(instr)
    run._r.append(fldEnd)
    set_run_font(run, size=8.5, color=MUTED)


def setup_document():
    doc = Document()
    sec = doc.sections[0]
    sec.page_width = Cm(21.0)
    sec.page_height = Cm(29.7)
    sec.top_margin = Cm(2.4)
    sec.bottom_margin = Cm(2.2)
    sec.left_margin = Cm(2.6)
    sec.right_margin = Cm(2.6)

    normal = doc.styles["Normal"]
    normal.font.name = FONT
    normal.font.size = Pt(10.5)
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    normal.paragraph_format.line_spacing = 1.6
    normal.paragraph_format.space_after = Pt(4)

    # 页眉
    header = sec.header
    hp = header.paragraphs[0]
    hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = hp.add_run("Touhou_RL · 研究路线复盘 · 2026-11")
    set_run_font(r, size=8.0, color=MUTED)

    # 页脚
    footer = sec.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_number(fp)

    doc.core_properties.title = "被证否推着走：Touhou_RL 三十七天研究路线复盘"
    doc.core_properties.author = "Touhou_RL 项目"
    return doc


def add_body(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.first_line_indent = Cm(0.74)
    p.paragraph_format.line_spacing = 1.65
    p.paragraph_format.space_after = Pt(5)
    add_inline(p, text)


def add_heading(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(7)
    p.paragraph_format.keep_with_next = True
    r = p.add_run(text)
    set_run_font(r, size=14.5, color=INK, bold=True)
    add_bottom_border(p, color="E6DFD1", size="4")
    return p


def add_image(doc, path):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(10)
    p.paragraph_format.keep_with_next = True
    # 行距必须复位为单倍，否则部分渲染器会按正文行高裁掉行内图片的底部
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.SINGLE
    run = p.add_run()
    run.add_picture(path, width=Cm(15.6))


def add_bullet(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.9)
    p.paragraph_format.first_line_indent = Cm(-0.45)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.45
    mark = p.add_run("· ")
    set_run_font(mark, size=10.0, color=MUTED, bold=True)
    add_inline(p, text, size=10.0)


def add_quote(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.6)
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.45
    add_inline(p, text, size=9.3, color=MUTED)


def add_rule(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    add_bottom_border(p, color="D8D3C8", size="6")


def main():
    with open(SRC, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    doc = setup_document()
    first_h1 = True
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("# "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            r = p.add_run(line[2:].strip())
            set_run_font(r, size=26, color=INK, bold=True)
            first_h1 = False
        elif line.startswith("## "):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(16)
            p.paragraph_format.keep_with_next = True
            r = p.add_run(line[3:].strip())
            set_run_font(r, size=12.0, color=MUTED)
            add_bottom_border(p, color="AD431F", size="8")
        elif line.startswith("### "):
            add_heading(doc, line[4:].strip())
        elif line.startswith("!["):
            m = re.match(r"!\[[^\]]*\]\(([^)]+)\)", line)
            if m:
                img_path = os.path.normpath(os.path.join(ROOT, m.group(1)))
                add_image(doc, img_path)
        elif line.startswith("- "):
            add_bullet(doc, line[2:].strip())
        elif line.startswith("> "):
            add_quote(doc, line[2:].strip())
        elif line.strip() == "---":
            add_rule(doc)
        else:
            add_body(doc, line.strip())

    doc.save(OUT)
    print("saved", OUT, os.path.getsize(OUT), "bytes")


if __name__ == "__main__":
    main()
