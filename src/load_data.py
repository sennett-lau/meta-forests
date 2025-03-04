import deeplake

def load_pacs_training_dataset():
  ds = deeplake.query('SELECT * FROM "hub://activeloop/pacs-train"')
  return ds.pytorch()
