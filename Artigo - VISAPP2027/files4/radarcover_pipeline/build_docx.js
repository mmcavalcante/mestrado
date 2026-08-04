const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  ImageRun, PageBreak, LevelFormat, convertInchesToTwip,
} = require('docx');

const BASE = '/home/claude/radarcover';
const OUT = path.join(BASE, 'RadarCover_Artigo_Completo.docx');

const files = [
  'capitulo_0_rosto.md',
  'capitulo_1_introducao.md',
  'capitulo_2_background.md',
  'capitulo_3_metodo_proposto.md',
  'capitulo_4_setup_experimental.md',
  'capitulo_5_resultados.md',
  'capitulo_6_conclusao.md',
];

// ---------- PNG dimension reader (parses IHDR chunk directly, no deps) ----------
function readPngDimensions(filePath) {
  const buf = fs.readFileSync(filePath);
  // PNG signature (8 bytes) + length(4) + 'IHDR'(4) + width(4) + height(4)
  const width = buf.readUInt32BE(16);
  const height = buf.readUInt32BE(20);
  return { width, height };
}

// ---------- inline formatting: bold / italic / inline-math -> TextRun[] ----------
function parseInline(text, extra = {}) {
  // normalize math delimiters to a lightweight readable form (no LaTeX engine available)
  text = text.replace(/\$\$(.+?)\$\$/g, (m, expr) => `[[MATH:${expr}]]`);
  text = text.replace(/\$(.+?)\$/g, (m, expr) => `[[MATH:${expr}]]`);

  const runs = [];
  // tokenize on **bold**, *italic*, [[MATH:...]]
  const tokenRe = /(\*\*.+?\*\*|\*[^*\s][^*]*?\*|\[\[MATH:.+?\]\])/g;
  let lastIndex = 0;
  let match;
  while ((match = tokenRe.exec(text)) !== null) {
    if (match.index > lastIndex) {
      runs.push(...splitMathTokens(text.slice(lastIndex, match.index), extra));
    }
    const tok = match[0];
    if (tok.startsWith('**')) {
      // bold may itself contain [[MATH:...]] placeholders (e.g. "**...$K{=}4$...**") —
      // split recursively instead of treating the inner text as an opaque string.
      runs.push(...splitMathTokens(tok.slice(2, -2), { ...extra, bold: true }));
    } else if (tok.startsWith('[[MATH:')) {
      const expr = tok.slice(7, -2);
      runs.push(new TextRun({ text: mathToText(expr), italics: true, font: 'Cambria Math', ...extra }));
    } else if (tok.startsWith('*')) {
      runs.push(...splitMathTokens(tok.slice(1, -1), { ...extra, italics: true }));
    }
    lastIndex = tokenRe.lastIndex;
  }
  if (lastIndex < text.length) {
    runs.push(...splitMathTokens(text.slice(lastIndex), extra));
  }
  return runs.length ? runs : [new TextRun({ text, ...extra })];
}

// Splits plain text (no ** or single-* markers expected) on any remaining
// [[MATH:...]] placeholders, producing TextRuns that inherit `extra`
// (bold/italics/size passed down from an enclosing token, if any).
function splitMathTokens(text, extra = {}) {
  const runs = [];
  const mathRe = /\[\[MATH:(.+?)\]\]/g;
  let lastIndex = 0;
  let m;
  while ((m = mathRe.exec(text)) !== null) {
    if (m.index > lastIndex) {
      runs.push(new TextRun({ text: text.slice(lastIndex, m.index), ...extra }));
    }
    runs.push(new TextRun({ text: mathToText(m[1]), font: 'Cambria Math', ...extra, italics: true }));
    lastIndex = mathRe.lastIndex;
  }
  if (lastIndex < text.length) {
    runs.push(new TextRun({ text: text.slice(lastIndex), ...extra }));
  }
  return runs;
}

// Finds the substring inside the FIRST balanced {...} group starting at
// or after `fromIdx`, returning {content, endIdx} (endIdx = index right
// after the closing brace). Needed because \frac{...}{...} arguments can
// themselves contain braces (e.g. \frac{\overline{\text{PSNR}}_j}{...}),
// which a single non-recursive regex cannot match correctly.
function extractBraceGroup(str, fromIdx) {
  let i = str.indexOf('{', fromIdx);
  if (i === -1) return null;
  let depth = 0;
  for (let j = i; j < str.length; j++) {
    if (str[j] === '{') depth++;
    else if (str[j] === '}') {
      depth--;
      if (depth === 0) return { content: str.slice(i + 1, j), endIdx: j + 1 };
    }
  }
  return null;
}

// Repeatedly finds and replaces \frac{A}{B} -> (A)/(B), innermost-safe
// because it re-scans from the start after every replacement.
function expandFrac(text) {
  let out = text;
  let idx;
  while ((idx = out.indexOf('\\frac')) !== -1) {
    const num = extractBraceGroup(out, idx + 5);
    if (!num) break;
    const den = extractBraceGroup(out, num.endIdx);
    if (!den) break;
    const replacement = `(${num.content})/(${den.content})`;
    out = out.slice(0, idx) + replacement + out.slice(den.endIdx);
  }
  return out;
}

function mathToText(expr) {
  expr = expandFrac(expr);
  return expr
    .replace(/\\left/g, '')
    .replace(/\\right/g, '')
    .replace(/\\sqrt\{([^{}]*)\}/g, '√($1)')
    .replace(/\\alpha/g, 'α')
    .replace(/\\beta/g, 'β')
    .replace(/\\chi/g, 'χ')
    .replace(/\\tau/g, 'τ')
    .replace(/\\sigma/g, 'σ')
    .replace(/\\pi/g, 'π')
    .replace(/\\ell/g, 'ℓ')
    .replace(/\\approx/g, '≈')
    .replace(/\\mapsto/g, '↦')
    .replace(/\\subseteq/g, '⊆')
    .replace(/\\bigcup/g, '⋃')
    .replace(/\\mathcal\{([^}]*)\}/g, '$1')
    .replace(/\\mathbf\{([^}]*)\}/g, '$1')
    .replace(/\\mathbb\{([^}]*)\}/g, (m, c) => (c === '1' ? '𝟙' : c))
    .replace(/\\text\{([^}]*)\}/g, '$1')
    .replace(/\\overline\{([^}]*)\}/g, '$1̄')
    .replace(/\\cdot/g, '·')
    .replace(/\\geq/g, '≥')
    .replace(/\\leq/g, '≤')
    .replace(/\\times/g, '×')
    .replace(/\\in/g, '∈')
    .replace(/\\forall/g, '∀')
    .replace(/\\wedge/g, '∧')
    .replace(/\\sum_\{([^}]*)\}\^\{([^}]*)\}/g, 'Σ($1→$2)')
    .replace(/\\sum_\{([^}]*)\}/g, 'Σ($1)')
    .replace(/\\sum/g, 'Σ')
    .replace(/\\max_?/g, 'max')
    .replace(/\\dots/g, '…')
    .replace(/\\qquad/g, '     ')
    .replace(/\\quad/g, '   ')
    .replace(/\\;/g, ' ')
    .replace(/\\,/g, ' ')
    .replace(/\\hat\{([^}]*)\}/g, '$1̂')
    .replace(/\\begin\{aligned\}/g, '')
    .replace(/\\end\{aligned\}/g, '')
    .replace(/\\\{/g, '§OB§')
    .replace(/\\\}/g, '§CB§')
    .replace(/\\\\/g, '   ;   ')
    .replace(/[{}]/g, '')
    .replace(/§OB§/g, '{')
    .replace(/§CB§/g, '}')
    .trim();
}

// ---------- shared line-wrapping helper ----------
// A markdown source line starts a NEW block (not a continuation of the
// current paragraph/list-item) if it matches any of these. Used uniformly
// by bullets, numbered lists, and plain paragraphs so that soft-wrapped
// list items (common when hand-editing .md at ~70-col width) are joined
// back into one logical item instead of being truncated mid-sentence.
function isBlockBoundary(line) {
  return (
    /^#{1,2} /.test(line) || /^\|/.test(line) || /^[-*] /.test(line) ||
    /^\d+\.\s/.test(line) || /^!\[/.test(line) || /^---+\s*$/.test(line) ||
    /^\$\$\s*$/.test(line.trim())
  );
}

function consumeWrappedLines(lines, startIdx) {
  const buf = [lines[startIdx].trim()];
  let i = startIdx + 1;
  while (i < lines.length && lines[i].trim() !== '' && !isBlockBoundary(lines[i])) {
    buf.push(lines[i].trim());
    i++;
  }
  return { text: buf.join(' '), nextIndex: i };
}

// ---------- block-level parser ----------
function parseMarkdownFile(filePath, chapterIdx) {
  const raw = fs.readFileSync(filePath, 'utf-8');
  const lines = raw.split('\n');
  const children = [];
  let i = 0;

  function pushParagraph(text, opts = {}) {
    children.push(new Paragraph({
      children: parseInline(text),
      spacing: { after: 160 },
      ...opts,
    }));
  }

  while (i < lines.length) {
    let line = lines[i];

    if (line.trim() === '') { i++; continue; }

    // Heading 1 (chapter)
    if (/^# /.test(line)) {
      const text = line.replace(/^# /, '').trim();
      children.push(new Paragraph({
        children: [new TextRun({ text, bold: true })],
        heading: HeadingLevel.HEADING_1,
        spacing: { before: 400, after: 240 },
        pageBreakBefore: chapterIdx > 0,
      }));
      i++; continue;
    }

    // Heading 2 (section)
    if (/^## /.test(line)) {
      const text = line.replace(/^## /, '').trim();
      children.push(new Paragraph({
        children: [new TextRun({ text, bold: true })],
        heading: HeadingLevel.HEADING_2,
        spacing: { before: 300, after: 160 },
      }));
      i++; continue;
    }

    // Horizontal rule
    if (/^---+\s*$/.test(line)) {
      children.push(new Paragraph({
        text: '', border: { bottom: { color: 'AAAAAA', space: 1, style: BorderStyle.SINGLE, size: 6 } },
        spacing: { after: 200 },
      }));
      i++; continue;
    }

    // Image
    const imgMatch = line.match(/^!\[(.*?)\]\((.*?)\)$/);
    if (imgMatch) {
      const alt = imgMatch[1];
      const imgPath = path.join(BASE, imgMatch[2]);
      if (fs.existsSync(imgPath)) {
        const dim = readPngDimensions(imgPath);
        const maxW = 560;
        const w = Math.min(maxW, dim.width);
        const h = Math.round((w / dim.width) * dim.height);
        children.push(new Paragraph({
          children: [new ImageRun({ type: 'png', data: fs.readFileSync(imgPath), transformation: { width: w, height: h } })],
          alignment: AlignmentType.CENTER,
          spacing: { before: 120, after: 60 },
        }));
        children.push(new Paragraph({
          children: [new TextRun({ text: alt, italics: true, size: 18 })],
          alignment: AlignmentType.CENTER,
          spacing: { after: 200 },
        }));
      }
      i++; continue;
    }

    // Table block
    if (/^\|/.test(line)) {
      const tblLines = [];
      while (i < lines.length && /^\|/.test(lines[i])) {
        tblLines.push(lines[i]);
        i++;
      }
      // remove separator row (---)
      const rows = tblLines.filter(l => !/^\|[\s:|-]+\|$/.test(l));
      const parsedRows = rows.map(l => l.split('|').slice(1, -1).map(c => c.trim()));
      const nCols = parsedRows[0].length;
      const tableWidthDXA = 9026;
      let colWidths;
      if (nCols > 3) {
        const firstColWidth = Math.floor(tableWidthDXA * 0.22);
        const restWidth = Math.floor((tableWidthDXA - firstColWidth) / (nCols - 1));
        colWidths = [firstColWidth, ...Array(nCols - 1).fill(restWidth)];
      } else {
        colWidths = Array(nCols).fill(Math.floor(tableWidthDXA / nCols));
      }
      const tableRows = parsedRows.map((cells, rIdx) => new TableRow({
        children: cells.map((cellText, cIdx) => new TableCell({
          width: { size: colWidths[cIdx], type: WidthType.DXA },
          shading: rIdx === 0 ? { type: ShadingType.CLEAR, fill: 'DDE6F0' } : undefined,
          children: [new Paragraph({
            children: parseInline(cellText, rIdx === 0 ? { bold: true, size: 18 } : { size: 18 }),
          })],
        })),
      }));
      children.push(new Table({
        rows: tableRows,
        width: { size: tableWidthDXA, type: WidthType.DXA },
        columnWidths: colWidths,
      }));
      children.push(new Paragraph({ text: '', spacing: { after: 200 } }));
      continue;
    }

    // Bullet list
    if (/^[-*] /.test(line)) {
      const { text: rawText, nextIndex } = consumeWrappedLines(lines, i);
      const text = rawText.replace(/^[-*] /, '');
      children.push(new Paragraph({
        children: parseInline(text),
        bullet: { level: 0 },
        spacing: { after: 80 },
      }));
      i = nextIndex; continue;
    }

    // Numbered list
    const numMatch = line.match(/^(\d+)\.\s+(.*)$/);
    if (numMatch) {
      const { text: rawText, nextIndex } = consumeWrappedLines(lines, i);
      const itemText = rawText.replace(/^\d+\.\s+/, '');
      children.push(new Paragraph({
        children: parseInline(itemText),
        numbering: { reference: 'default-numbering', level: 0 },
        spacing: { after: 80 },
      }));
      i = nextIndex; continue;
    }

    // Blockquote / italic standalone (used for figure notes at end of ch2)
    if (/^\*[^*].*\*$/.test(line.trim())) {
      const text = line.trim().slice(1, -1);
      children.push(new Paragraph({
        children: [new TextRun({ text, italics: true, size: 19 })],
        spacing: { after: 160 },
      }));
      i++; continue;
    }

    // Display math block ($$ on its own line ... $$ on its own line)
    if (/^\$\$\s*$/.test(line.trim())) {
      const mathLines = [];
      i++;
      while (i < lines.length && !/^\$\$\s*$/.test(lines[i].trim())) {
        if (lines[i].trim() !== '') mathLines.push(lines[i].trim());
        i++;
      }
      i++; // skip closing $$
      const expr = mathLines.join(' ');
      children.push(new Paragraph({
        children: [new TextRun({ text: mathToText(expr), italics: true, font: 'Cambria Math' })],
        alignment: AlignmentType.CENTER,
        spacing: { before: 160, after: 200 },
      }));
      continue;
    }

    // Default: paragraph (accumulate wrapped lines until blank/next block)
    const { text: paraText, nextIndex } = consumeWrappedLines(lines, i);
    i = nextIndex;
    pushParagraph(paraText);
  }

  return children;
}

// ---------- build document ----------
let allChildren = [];
files.forEach((f, idx) => {
  const chunk = parseMarkdownFile(path.join(BASE, f), idx);
  allChildren = allChildren.concat(chunk);
});

const doc = new Document({
  numbering: {
    config: [{
      reference: 'default-numbering',
      levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.START,
        style: { paragraph: { indent: { left: convertInchesToTwip(0.35), hanging: convertInchesToTwip(0.25) } } } }],
    }],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 11906, height: 16838 }, // A4
        margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 },
      },
    },
    children: allChildren,
  }],
  styles: {
    default: {
      document: { run: { font: 'Calibri', size: 22 } }, // 11pt
    },
  },
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(OUT, buffer);
  console.log('Wrote', OUT, buffer.length, 'bytes');
});
