from acoss.utils import COVERS_80_CSV
from acoss.extractors import batch_feature_extractor
from acoss.extractors import PROFILE

extractor_profile = {
           'sample_rate': 32000,
           'input_audio_format': '.mp3',
           'downsample_audio': True,
           'downsample_factor': 2,
           'endtime': None,
           'features': ['hpcp', 'key_extractor', 'madmom_features', 'mfcc_htk']
}

audio_dir = "/dataset/coversongs/covers32k/"
feature_dir = "/dataset/coversongs/features/"
batch_feature_extractor(dataset_csv=COVERS_80_CSV,
                        audio_dir=audio_dir,
                        feature_dir=feature_dir,
                        n_workers=1,
                        mode="parallel",
                        params=extractor_profile)
