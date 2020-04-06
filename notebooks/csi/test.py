from pipeline import *

files = [
  '/dataset/YTCdataset/letitbe/v0.mp3',
  '/dataset/YTCdataset/letitbe/v1.mp3',
  '/dataset/YTCdataset/jbg/v2.mp3',
  '/dataset/YTCdataset/jbg/v3.mp3'
]
#files = ['/dataset/YTCdataset/letitbe/test.mp3']
multipliers = [1, 2, 3, 4, 8, 12, 16, 32]
ssm_params = {'width':1, 'mode':'affinity', 'sym':True}
#checkerboard = profiles.SSMCheckerboardBox(**ssm_params)

p = Pipeline()
p.use_feature(features.ChromagramCENS(hop_length=2**12))
#p.create_profiles(checkerboard, multipliers)
p.create_profiles(profiles.SimpleFast(), multipliers)
p.detect_peaks(peak_detection.FindPeaks())
p.tokenize(tokenizers.NearestChildrenSegment(length=3, connections=3, n_bits=8))
p.run(files)


def matches(song1, song2):
  s1 = set(song1.tokens)
  s2 = set(song2.tokens)
  return len([value for value in s1 if value in s2])


for song1 in p.songs:
  for song2 in p.songs:
    m = matches(song1, song2)
    s1 = set(song1.tokens)
    s2 = set(song2.tokens)
    print(song1.path, song2.path, m, m/len(s1), m/len(s2))

