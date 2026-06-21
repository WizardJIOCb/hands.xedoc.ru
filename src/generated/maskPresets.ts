export type MaskPreset = {
  id: string
  label: string
  url: string
}

export const maskPresets = [
  {
    "id": "joker",
    "label": "Joker",
    "url": "/masks/Joker.jpg"
  },
  {
    "id": "lukashenko",
    "label": "Lukashenko",
    "url": "/masks/Lukashenko.jpg"
  }
] satisfies MaskPreset[]
