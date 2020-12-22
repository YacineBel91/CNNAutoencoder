import hashlib
import imghdr
import os


def listImagesInDir(directory, ignoreDirs=[]):
    res = []
    ignoreDirs = [d.split("/")[-1] for d in ignoreDirs]
    for root, dirs, files in os.walk(directory, topdown=True):
        dirs[:] = [d for d in set(dirs) - set(ignoreDirs)]
        for fileName in files:
            if imghdr.what(os.path.join(root, fileName)) in ['rgb', 'gif', 'pbm', 'pgm', 'ppm', 'tiff', 'xbm', 'jpeg',
                                                             'bmp', 'png', 'webp']:
                res.append(os.path.join(root, fileName))
    return res


def listMD5inDir(directory, ignoreDirs=[]):
    allFileNames = listImagesInDir(directory, ignoreDirs)
    allSums = []
    for fn in allFileNames:
        hasher = hashlib.md5()
        hasher.update(open(fn, "rb").read())
        allSums.append(hasher.hexdigest())
    return allSums


def checkDifferencesBetweenTwoFolders(folder1, folder2):
    sums1 = listMD5inDir(folder1)
    sums2 = listMD5inDir(folder2)
    return set(sums1) - set(sums2)
