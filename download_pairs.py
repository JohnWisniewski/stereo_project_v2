import urllib.request, os, pathlib, ssl
ssl._create_default_https_context = ssl._create_unverified_context  # skip some Win-SSL nags
pairs = {
    "cones": ("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im2.png",
              "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/cones/im6.png"),
    "teddy": ("https://vision.middlebury.edu/stereo/data/scenes2003/newdata/teddy/im2.png",
              "https://vision.middlebury.edu/stereo/data/scenes2003/newdata/teddy/im6.png"),
  "tsukuba": ("https://raw.githubusercontent.com/1kc2/Disparity-Map/main/Stereo%20Pairs/Pair%201/view1.png",
              "https://raw.githubusercontent.com/1kc2/Disparity-Map/main/Stereo%20Pairs/Pair%201/view2.png"),
}
os.makedirs("bench_pairs", exist_ok=True)
for name,(L,R) in pairs.items():
    for url,tag in [(L,"left"),(R,"right")]:
        dst = pathlib.Path("bench_pairs")/f"{name}_{tag}.png"
        urllib.request.urlretrieve(url, dst)
        print("✓", dst)
print("\nAll images saved in bench_pairs\\")
