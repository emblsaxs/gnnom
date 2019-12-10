import sys
import os
import fnmatch
import shutil

# python copy_from_file (input_root_directory) (output__directory)
filePath = sys.argv[1]
outPath  = os.path.join(os.getcwd(),sys.argv[2])

with open(filePath, 'r') as f:
    lines = f.readlines()
lines = [x.strip() for x in lines]

os.mkdir(outPath)

os.mkdir(os.path.join(outPath, "ints"))

os.mkdir(os.path.join(outPath, "logs"))

for line in lines:
  for root, dirnames, filenames in os.walk('/data/local/intfiles'):
    for filename in fnmatch.filter(filenames, line + '.int'):
      f = os.path.join(root,filename)
      print("New file is being copied... " + f)
      shutil.copy(f, outPath+ "/ints")
    for filename in fnmatch.filter(filenames, line + '.log'):
      f = os.path.join(root,filename)
      print("New file is being copied... " + f)
      shutil.copy(f, outPath+ "/logs")

print("Well done!")
print(str(len(lines)) + "files are copied.")
