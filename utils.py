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

def get_ax_variations_names():
  variations_names = []
  for yaw in range(-180, 180 + 1):
      for pitch in range(-180, 180 + 1):
          for roll in range(-180, 180 + 1):
              variations_names.append(f"y{yaw}-p{pitch}-r{roll}")
  return variations_names

