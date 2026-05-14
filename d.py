import xarray as xr

ds = xr.open_dataset(r"C:\Users\srikar\OneDrive\Desktop\all projects\digitaltwins\data\rainfall\chirps-v2.0.2018.days_p05.nc")

rain = ds.sel(latitude=25, longitude=90, method="nearest")["precip"]

print(rain)
june = rain.sel(time="2018-06")

for t, val in zip(june.time.values, june.values):
    print(t, ":", val, "mm")

print("June Avg:", june.mean().values)
print("June Max:", june.max().values)
print("June Total:", june.sum().values)