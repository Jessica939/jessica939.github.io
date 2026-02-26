/**
 * 自动扫描markdown文件，提取文件修改时间并添加到Front Matter
 * 使用方法: node scripts/updateFileMetadata.js
 */

const fs = require('fs')
const path = require('path')

const NOTES_DIR = path.join(__dirname, '../notes')
const IGNORE_DIRS = ['.vitepress', 'public', 'unused']

function formatDate(date) {
  return date.toISOString().split('T')[0] // YYYY-MM-DD 格式
}

function extractFrontMatter(content) {
  const frontMatterRegex = /^---\n([\s\S]*?)\n---\n([\s\S]*)$/
  const match = content.match(frontMatterRegex)
  
  if (match) {
    const frontMatter = match[1]
    const body = match[2]
    return { frontMatter, body, hasFrontMatter: true }
  }
  
  return { frontMatter: '', body: content, hasFrontMatter: false }
}

function parseFrontMatter(frontMatterStr) {
  const lines = frontMatterStr.split('\n')
  const metadata = {}
  
  lines.forEach(line => {
    const match = line.match(/^(\w+):\s*(.+)$/)
    if (match) {
      metadata[match[1].toLowerCase()] = match[2].trim().replace(/^['"]|['"]$/g, '')
    }
  })
  
  return metadata
}

function stringifyFrontMatter(metadata) {
  return Object.entries(metadata)
    .map(([key, value]) => `${key}: ${value}`)
    .join('\n')
}

function processFile(filePath) {
  try {
    const stats = fs.statSync(filePath)
    const content = fs.readFileSync(filePath, 'utf-8')
    const { frontMatter, body, hasFrontMatter } = extractFrontMatter(content)
    
    const metadata = hasFrontMatter ? parseFrontMatter(frontMatter) : {}
    const modifiedDate = formatDate(stats.mtime)
    
    // 如果没有 date 字段，或者需要更新，就添加/更新它
    if (!metadata.date) {
      metadata.date = modifiedDate
      
      const updatedFrontMatter = stringifyFrontMatter(metadata)
      const newContent = `---\n${updatedFrontMatter}\n---\n${body}`
      
      fs.writeFileSync(filePath, newContent, 'utf-8')
      console.log(`✅ 已更新: ${filePath} (日期: ${modifiedDate})`)
      return true
    } else {
      console.log(`⏭️ 跳过 (已有日期): ${filePath} (日期: ${metadata.date})`)
      return false
    }
  } catch (error) {
    console.error(`❌ 错误: ${filePath} - ${error.message}`)
    return false
  }
}

function walkDir(dir) {
  const files = fs.readdirSync(dir)
  
  files.forEach(file => {
    const filePath = path.join(dir, file)
    const stat = fs.statSync(filePath)
    
    if (stat.isDirectory()) {
      if (!IGNORE_DIRS.includes(file)) {
        walkDir(filePath)
      }
    } else if (file.endsWith('.md') && file !== 'index.md') {
      processFile(filePath)
    }
  })
}

console.log('🚀 开始扫描markdown文件...\n')
walkDir(NOTES_DIR)
console.log('\n✨ 完成！')
