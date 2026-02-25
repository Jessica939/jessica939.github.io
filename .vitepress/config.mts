import { defineConfig } from 'vitepress'
import mathjax3 from 'markdown-it-mathjax3'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "Jessica's homepage",
  description: "welcome",
  base: '/', 
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/' },
      { text: '我的笔记', link: '/notes/' },
      { text: 'Some Ideas', link: '/ideas/' }
    ],

    sidebar: {
      '/notes/AIInfra': [
        {
          text: 'AI Infra',
          items: [
            { text: 'How to Train Really Large Models', link: '/notes/AIInfra/How_to_Train_Really_Large_Models_on_Many_GPUs/reading notes'},
            { text:'The Illustrated Transformer', link:'/notes/AIInfra/The_Illustrated_Transformer/note.md'},
            { text:'Large Transformer Model Inference Optimization', link:'notes/AIInfra/Large_Transformer_Model_Inference_Optimization/note.md'}
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/Jessica939' }
    ],
    outline: {
      level: [2, 4], // 意思是显示 h2 和 h3 级别的标题
      label: '页面导航' // 右侧栏的标题文本，默认是 "On this page"
    },
  },
  markdown: {
    config: (md) => {
      md.use(mathjax3) // 启用 MathJax 支持
    }
  },
  
})
