import { defineConfig } from "blume";

const inter = {
  name: "Inter",
  variants: [{ src: "../node_modules/@fontsource-variable/inter/files/inter-latin-wght-normal.woff2" }],
};

const products = [
  { label: "评测框架", path: "/evaluate" },
  { label: "插件开发", path: "/plugins" },
];

export default defineConfig({
  title: "TallyJudge",
  description: "用 TallyJudge 评测 RAG 与 Agent 应用，开发自定义指标与模型适配器。",
  content: { root: ".", exclude: ["public/**"] },
  deployment: { site: process.env.DOCS_SITE, base: process.env.DOCS_BASE || "/" },
  logo: { image: "/icon.png", text: "TallyJudge" },
  i18n: {
    defaultLocale: "zh-CN",
    locales: [{ code: "zh-CN", label: "简体中文" }],
  },
  navigation: {
    // Tabs scope the page tree and pagination; the header dropdown switches products.
    tabs: products,
    sidebar: { display: "flat" },
    selectors: [
      {
        kind: "product",
        label: "文档区",
        items: [{ label: "文档首页", path: "/" }, ...products],
      },
    ],
  },
  theme: {
    fonts: {
      display: inter,
      body: inter,
      mono: {
        name: "IBM Plex Mono",
        variants: [
          { src: "../node_modules/@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-400-normal.woff2", weight: 400 },
          { src: "../node_modules/@fontsource/ibm-plex-mono/files/ibm-plex-mono-latin-600-normal.woff2", weight: 600 },
        ],
      },
    },
  },
  search: { indexing: { includeCodeBlocks: true } },
  markdown: { code: { wrap: true } },
  toc: { minHeadingLevel: 2, maxHeadingLevel: 2 },
  github: { owner: "datawhalechina", repo: "tally-judge", dir: "docs" },
  footer: {
    links: [{ label: "报告文档问题", href: "https://github.com/datawhalechina/tally-judge/issues" }],
  },
  seo: { og: { enabled: false } },
});
