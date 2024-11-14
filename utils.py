def get_variations_names():
  variations_names = []
  cameras = ["c00", "c02", "c04", "c07", "c09"]
  rotations = ["r01", "r02", "r03", "r04", "r06", "r07"]
  lighting = "l0"
  focus_value = "f2"
  for camera in cameras:
      for rotation in rotations:
          variations_names.append(f"{camera}-{rotation}-{lighting}-{focus_value}")
  return variations_names