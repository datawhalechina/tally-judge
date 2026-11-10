# 导出 PDF { #pdf-export }

## 用途 { #purpose }
PDF 导出功能通过 MkDocs 和 `mkdocs-to-pdf` 插件，将完整的 Ragas 文档生成为一个 PDF 文件。

## 用法 { #usage }

实现中使用两份 MkDocs 配置。
- `mkdocs.yml` 用于常规 HTML 构建，无须 PDF 相关依赖
- `mkdocs-pdf.yml` 继承主配置，并添加 PDF 插件

构建 PDF 文档。
```bash
make build-docs-pdf
```

生成的 PDF 位于 `site/pdf/document.pdf`。

仅构建 HTML 文档。
```bash
make build-docs
```

`make build-docs-pdf` 命令会在构建前自动检查系统依赖。

## PDF 中的 Mermaid 图表（离线） { #mermaid-diagrams-in-pdf-offline }
构建 PDF 时，Mermaid 图表会**离线**渲染，在 WeasyPrint 运行之前转换为 SVG。除 WeasyPrint 外，还需要以下依赖。

### 必需工具 { #required-tools }
- Node.js，用于运行 Mermaid 工具
- Mermaid CLI（`mmdc`），通过 `@mermaid-js/mermaid-cli` 安装
- Puppeteer 使用的无头浏览器，推荐 `chrome-headless-shell`


## 当前限制 { #current-limitations }

**系统依赖**。WeasyPrint 需要额外安装操作系统对应的系统库，例如 Pango 和 Cairo。如遇问题，请参阅 [WeasyPrint 安装说明](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html)与[故障排查指南](https://doc.courtbouillon.org/weasyprint/stable/first_steps.html#troubleshooting)。

**ReadTheDocs**。当前 ReadTheDocs 构建配置尚未启用 PDF 生成。
