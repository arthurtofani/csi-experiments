from heapq import nsmallest
from bisect import bisect_right
import base64
from util import *

class Tokenizer:
  def __init__(self, **kwargs):
    self._kwargs = kwargs

  def execute(self, song):
    return self._exec(song)

  def _exec(self, song):
    raise NotImplementedError


class NearestChildrenSegment(Tokenizer):

  def _exec(self, song):
    params = {'length':2, 'connections':2, 'n_bits': 4}
    params.update(self._kwargs)
    peaks = song.peaks[::-1]
    paths = self.get_paths(peaks, params['length'], params['connections'], 0)
    return self.normalize_paths(paths, params['n_bits'])

  def normalize_paths(self, paths, n_bits):
    def quant(n):
      return bit_quantize(n, n_bits)
    arr = []
    for path in paths:
      r = normalize(path)[1:-1]
      ll = list(map(quant, r))
      arr.append(base64.b64encode(bytes(ll)))
    return arr

  def get_paths(self, peaks, length, connections, iteration):
    results = []
    for i in range(len(peaks))[:-(length)]:
      line_peaks = [0] if len(peaks[i])==0 else peaks[i]
      for peak in line_peaks:
        self.walk_in_depth(peak, peaks, i, length, connections, 0, results, [peak])
    return results

  def walk_in_depth(self, current_peak, peaks, line_idx, length, connections, iteration, arr_results, tmp=[]):
    nearest_children = self.k_nearest(connections, current_peak, peaks[line_idx+1])
    for child in nearest_children:
      if iteration == length-1:
        arr_results.append(tmp + [child])
      else:
        self.walk_in_depth(child, peaks, line_idx+1, length, connections, iteration+1, arr_results, tmp + [child])

  def k_nearest(self, k, center, sorted_data):
    i = bisect_right(sorted_data, center)
    return sorted_data[i : i+k]
