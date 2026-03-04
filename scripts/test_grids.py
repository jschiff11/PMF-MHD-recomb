from pmhd.data.grids import k_grid

karr = k_grid()

print("Testing karr")
print("karr min:", karr.min())
print("karr max:", karr.max())
print("Number of points:", len(karr))

