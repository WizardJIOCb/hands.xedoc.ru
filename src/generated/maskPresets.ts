export type MaskPreset = {
  id: string
  label: string
  url: string
}

export const maskPresets = [
  {
    "id": "einstein-2",
    "label": "Einstein-2",
    "url": "/masks/Einstein-2.webp"
  },
  {
    "id": "einstein",
    "label": "Einstein",
    "url": "/masks/Einstein.jpg"
  },
  {
    "id": "joker",
    "label": "Joker",
    "url": "/masks/Joker.jpg"
  },
  {
    "id": "lukashenko",
    "label": "Lukashenko",
    "url": "/masks/Lukashenko.jpg"
  },
  {
    "id": "obama",
    "label": "Obama",
    "url": "/masks/Obama.jpg"
  },
  {
    "id": "trump",
    "label": "Trump",
    "url": "/masks/Trump.jpg"
  }
] satisfies MaskPreset[]
