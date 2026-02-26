import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Jessica's Homepage",
  description: "A space for ideas and notes",
  base: '/',
  lastUpdated: true,

  locales: {
    root: {
      label: '简体中文',
      lang: 'zh-CN',
      themeConfig: {
        nav: [
          { text: '🏠 home', link: '/' },
          { text: '📖 notes', link: '/notes/' },
          { text: '💡 ideas', link: '/ideas/' },
          { text: '🙋 about me', link: '/about' },
        ],
        socialLinks: [
          { icon: 'github', link: 'https://github.com/Jessica939' }
        ],
        outline: {
          level: [2, 4],
          label: '页面导航'
        },
        lastUpdated: {
          text: '最后更新于'
        }
      }
    },
    en: {
      label: 'English',
      lang: 'en-US',
      link: '/en/',
      themeConfig: {
        nav: [
          { text: '🏠 home', link: '/en/' },
          { text: '📖 notes', link: '/en/notes/' },
          { text: '💡 ideas', link: '/en/ideas/' },
          { text: '🙋 about me', link: '/en/about' },
          { text: '🐙 GitHub', link: 'https://github.com/Jessica939' }
        ],
        socialLinks: [
          { icon: 'github', link: 'https://github.com/Jessica939' }
        ],
        outline: {
          level: [2, 4],
          label: 'On this page'
        },
        lastUpdated: {
          text: 'Last updated'
        }
      }
    }
  },

  markdown: {
    config: (md) => {
      md.use(mathjax3)
    }
  },

  head: [
    ['link', { rel: 'icon', type: 'image/png', href: '/logo.png' }]
  ]
})
