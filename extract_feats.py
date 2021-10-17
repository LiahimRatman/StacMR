from vocab import Vocabulary
import evaluation


def extract_features(model_path='runs/model_best.pth.tar', data_path='data_big/', fold5=False):
    """
    Extracting features for search evaluation with model given in repo. It works for flickr30k dataset (it's included in it's options)
    As the result final embeddings for images and captions are constructed
    :param model_path: Path to VSRN trained model
    :param data_path: Path to data with precomputed embeddings for OCR, images and to captions (18Gb dataset)
    :param fold5:
    :return:
    """
    print('Extracting features for CTC SPLITS!')
    evaluation.extract_feats(model_path,
                             data_path=data_path,
                             split="dev",
                             fold5=fold5)
    evaluation.extract_feats(model_path,
                             data_path=data_path,
                             split="test",
                             fold5=fold5)


if __name__ == "__main__":
    extract_features()
