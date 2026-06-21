import { mkdirSync, readdirSync, writeFileSync } from 'node:fs'
import { extname, join, parse } from 'node:path'

const root = process.cwd()
const masksDir = join(root, 'public', 'masks')
const outputDir = join(root, 'src', 'generated')
const outputFile = join(outputDir, 'maskPresets.ts')
const imageExtensions = new Set(['.avif', '.gif', '.jpeg', '.jpg', '.png', '.webp'])

let files = []

try {
  files = readdirSync(masksDir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && imageExtensions.has(extname(entry.name).toLowerCase()))
    .map((entry) => entry.name)
    .sort((a, b) => a.localeCompare(b, 'ru'))
} catch {
  files = []
}

const presets = files.map((fileName) => ({
  id: parse(fileName).name.toLowerCase().replace(/[^a-z0-9а-яё]+/gi, '-').replace(/^-|-$/g, ''),
  label: parse(fileName).name,
  url: `/masks/${encodeURIComponent(fileName)}`,
}))

mkdirSync(outputDir, { recursive: true })
writeFileSync(
  outputFile,
  [
    'export type MaskPreset = {',
    '  id: string',
    '  label: string',
    '  url: string',
    '}',
    '',
    `export const maskPresets = ${JSON.stringify(presets, null, 2)} satisfies MaskPreset[]`,
    '',
  ].join('\n'),
)
