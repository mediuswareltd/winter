import { defineConfig } from 'vitepress'

export default defineConfig({
  title: "Winter",
  description: "Multimodal retrieval over speech and keyframes so every answer is cited by timestamp.",
  base: '/winter/',
  cleanUrls: true,
  lastUpdated: true,
  head: [
    ['link', { rel: 'icon', href: '/favicon.svg' }],
  ],
  themeConfig: {
    logo: { light: '/logo.svg', dark: '/logo.svg', alt: "Winter" },
    siteTitle: "Winter",
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Guide', link: '/guide' },
      { text: 'Getting started', link: '/getting-started' },
      {
        text: 'GitHub',
        link: 'https://github.com/mediuswareltd/winter'
      }
    ],
    sidebar: [
      {
        text: 'Documentation',
        items: [
          { text: 'Overview', link: '/' },
          { text: 'Getting started', link: '/getting-started' },
          { text: 'Guide', link: '/guide' },
          { text: 'About', link: '/about' }
        ]
      }
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/mediuswareltd/winter' }
    ],
    footer: {
      message: 'Built by Mediusware',
      copyright: 'Copyright © Mediusware Ltd'
    },
    editLink: {
      pattern: 'https://github.com/mediuswareltd/winter/edit/main/docs/:path',
      text: 'Edit this page on GitHub'
    },
    search: {
      provider: 'local'
    },
    outline: [2, 3]
  }
,
  vite: {
    css: {
      postcss: {
        plugins: []
      }
    }
  }
})
