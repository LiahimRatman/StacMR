import torch
from vocab import Vocabulary
import evaluation_models
import evaluation


# for flickr
def evaluate_models(model_path='runs/model_best.pth.tar', data_path='data_big/', split='dev', fold5=False):
    evaluation.evalrank(model_path=model_path,
                        data_path=data_path,
                        split=split,
                        fold5=fold5)


if __name__ == "__main__":
    print('\nEvaluation on Flickr30K:')
    evaluate_models(data_path='data_big/', split='dev')  # Для test нет данных

    # Все, что ниже не запускается из-за отсутствия данных

    # print('\nEvaluation on TextCaps Validation set:')
    # evaluate_models(data_path='data/', split='dev')
    #
    # print('\nEvaluation on STACMR:')
    # evaluate_models(data_path='data/', split='dev')
    # evaluate_models(data_path='data/', split='test')
