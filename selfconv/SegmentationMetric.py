"""
https://blog.csdn.net/sinat_29047129/article/details/103642140
https://www.cnblogs.com/Trevo/p/11795503.html
refer to https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
"""
import numpy as np
import os
from PIL import Image
import logging

__all__ = ['SegmentationMetric']

"""
confusionMetric
P\L     P    N
P      TP    FP
N      FN    TN
"""


def cvtType(img, List):
    H, W, C = img.shape
    img = img.reshape(H*W, C)
    img_cvt = np.zeros(H*W, dtype=img.dtype)
    for l in range(len(List)):
        idx = np.where((img[:, 0] == List[l][0]) & (img[:, 1] == List[l][1]) & (img[:, 2] == List[l][2]))
        img_cvt[idx] = l

    img_cvt = img_cvt.reshape(H, W)

    return img_cvt

class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)  # 混淆矩阵n*n，初始值全0


    # 像素准确率PA，预测正确的像素/总像素
    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    # 类别像素准确率CPA，返回n*1的值，代表每一类，包括背景,Precession==classPixelAccuracy
    # def classPixelAccuracy(self):
    #     # return each category pixel accuracy(A more accurate way to call it precision)
    #     # acc = (TP) / TP + FP
    #     # classAcc = np.diag(self.confusionMatrix) / np.maximum(self.confusionMatrix.sum(axis=1), 1)
    #     classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
    #     return classAcc
    def Precision(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        precision = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        return precision

    # 类别平均像素准确率MPA，对每一类的像素准确率求平均
    def meanPixelAccuracy(self):
        classAcc = self.Precision()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def Recall(self):
        # Recall = (TP) / (TP + FN)
        recall = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return recall

    def meanRecall(self):
        recall = self.Recall()
        mRecall = np.nanmean(recall)
        return mRecall

    def F1(self):
        # 2*precision*recall / (precision + recall)
        f1 = 2 * self.Precision() * self.Recall() / (self.Precision() + self.Recall())
        return f1

    def meanF1(self):
        f1 = self.F1()
        mF1 = np.nanmean(f1)
        return mF1
    # MIoU
    def IntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        # union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
        #     self.confusionMatrix)
        union = np.maximum(np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
                    self.confusionMatrix), 1)
        IoU = intersection / union
        return IoU

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        IoU = self.IntersectionOverUnion()
        mIoU = np.nanmean(IoU)
        return mIoU
    # def meanIntersectionOverUnion(self):
    #     # Intersection = TP Union = TP + FP + FN
    #     # IoU = TP / (TP + FP + FN)
    #     intersection = np.diag(self.confusionMatrix)
    #     union = np.maximum(np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
    #         self.confusionMatrix), 1)
    #     IoU = intersection / union
    #     mIoU = np.nanmean(IoU)
    #     return mIoU

    # 根据标签和预测图片返回其混淆矩阵
    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask].astype(int) + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusionMatrix, axis=1) / np.sum(self.confusionMatrix)
        iu = np.diag(self.confusionMatrix) / (
                np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) -
                np.diag(self.confusionMatrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    # add kappa
    def Kappa(self):
        pe_rows = np.sum(self.confusionMatrix, axis=0)
        pe_cols = np.sum(self.confusionMatrix, axis=1)
        sum_total = sum(pe_cols)
        pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
        po = np.trace(self.confusionMatrix) / float(sum_total)
        kappa = (po - pe) / (1 - pe)

        return kappa

    # 更新混淆矩阵
    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape  # 确认标签和预测值图片大小相等
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    # 清空混淆矩阵
    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


def old():
    imgPredict = np.array([0, 0, 0, 1, 2, 2])
    imgLabel = np.array([0, 0, 1, 1, 2, 2])
    metric = SegmentationMetric(3)
    metric.addBatch(imgPredict, imgLabel)
    acc = metric.pixelAccuracy()
    macc = metric.meanPixelAccuracy()
    mIoU = metric.meanIntersectionOverUnion()
    print(acc, macc, mIoU)


# def evaluate1(pre_path, label_path, pre_suffix='.jpg', label_suffix='.png'):
#     acc_list = []
#     macc_list = []
#     mIoU_list = []
#     fwIoU_list = []
#     kappa_list = []
#
#     pre_imgs = os.listdir(pre_path)
#     lab_imgs = os.listdir(label_path)
#
#     name_list = list(map(lambda x: x[:-4], os.listdir(label_path)))
#     for i, p in enumerate(name_list):
#
#         imgPredict = Image.open(pre_path + p + pre_suffix)
#         imgPredict = np.array(imgPredict)
#         imgPredict[imgPredict < 128] = 0
#         imgPredict[imgPredict > 0] = 255
#         List = [[255, 255, 255],[0, 0, 255],[0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
#         img_cvt = cvtType(imgPredict, List)
#
#         # imgPredict = imgPredict[:,:,0]
#         imgLabel = Image.open(label_path + p + label_suffix)
#         imgLabel = np.array(imgLabel)
#         # imgLabel = imgLabel[:,:,0]
#
#         metric = SegmentationMetric(6)  # 表示分类个数，包括背景
#         metric.addBatch(img_cvt, imgLabel)
#         acc = metric.pixelAccuracy()
#         macc = metric.meanPixelAccuracy()
#         mIoU = metric.meanIntersectionOverUnion()
#         fwIoU = metric.Frequency_Weighted_Intersection_over_Union()
#         kappa = metric.Kappa()
#
#         acc_list.append(acc)
#         macc_list.append(macc)
#         mIoU_list.append(mIoU)
#         fwIoU_list.append(fwIoU)
#         kappa_list.append(kappa)

    #     # print('{}: acc={}, macc={}, mIoU={}, fwIoU={}'.format(p, acc, macc, mIoU, fwIoU))
    #
    # return acc_list, macc_list, mIoU_list, fwIoU_list,  kappa_list


def evaluate2(pre_path, label_path, pre_suffix='.jpg', label_suffix='.png'):
    pre_imgs = os.listdir(pre_path)
    lab_imgs = os.listdir(label_path)

    name_list = list(map(lambda x: x[:-4], os.listdir(label_path)))

    metric = SegmentationMetric(6)  # 表示分类个数，包括背景
    for i, p in enumerate(name_list):
        imgPredict = Image.open(pre_path + p + pre_suffix)
        imgPredict = np.array(imgPredict)
        imgPredict[imgPredict < 128] = 0
        imgPredict[imgPredict > 0] = 255
        List = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
        img_cvt = cvtType(imgPredict, List)

        # imgPredict = imgPredict[:,:,0]
        imgLabel = Image.open(label_path + p + label_suffix)
        imgLabel = np.array(imgLabel)

        metric.addBatch(img_cvt, imgLabel)

        # imgPredict = Image.open(pre_path + p)
        # imgPredict = np.array(imgPredict)
        # imgLabel = Image.open(label_path + lab_imgs[i])
        # imgLabel = np.array(imgLabel)

        # metric.addBatch(imgPredict, imgLabel)

    return metric


if __name__ == '__main__':
    # pre_path = './pre_path/'
    # label_path = './label_path/'

    # # potsdam
    # pre_path = '/media/oyasumi/CECD0248603A9AB5/degeneration/potsdam/SwinTv2/RandomShadowTest/'
    # # logpre_path = '/media/oyasumi/CECD0248603A9AB5/W2result/potsdam_result/mixtransformer.b4.largehead/'
    # label_path = '/home/oyasumi/Documents/DATASEGMENTATION/new_potsdam/test/mask_pseudo/'
    #vaihingen
    pre_path = '/media/oyasumi/CECD0248603A9AB5/vai_out_base/acmixLawinV8/Test/'
    # logpre_path = '/media/oyasumi/CECD0248603A9AB5/W2result/vaihingen_result/mixtransformer.b4.largehead/'
    label_path = '/home/oyasumi/Documents/DATASEGMENTATION/new_vaihingen/test/mask_pseudo/'


    # 加总测试集每张图片的混淆矩阵，对最终形成的这一个矩阵计算各种评价指标
    metric = evaluate2(pre_path, label_path)
    IoU = metric.IntersectionOverUnion()
    mIoU = np.nanmean(IoU[0:5])
    acc = metric.pixelAccuracy()
    precision = metric.Precision()
    mprecision = np.nanmean(precision[0:5])
    macc = metric.meanPixelAccuracy()
    F1 = metric.F1()
    mF1 = np.nanmean(F1[0:5])
    recall = metric.Recall()
    mrecall = np.nanmean(recall[0:5])
    Kappa = metric.Kappa()

    print(pre_path + ' result:')

    # for i in range(len(IoU)):
    #     # print('|{}:IoU:{}% F1:{}% precision:{}% recall:{}%|'.format(str([i]).ljust(4), str(round(IoU[i], 4)).rjust(10),str(round(F1[i], 4)).rjust(10),
    #     #                              str(round(precision[i], 4)).rjust(10), str(round(recall[i], 4)).rjust(10),
    #     #                              ))
    #     print('|{}:IoU:{}% F1:{}% precision:{}% recall:{}%|'.format(str([i]).ljust(4), str(round(IoU[i]* 100, 2)).rjust(10),
    #                                                                 str(round(F1[i]* 100, 2)).rjust(10),
    #                                                                 str(round(precision[i]* 100, 2)).rjust(10),
    #                                                                 str(round(recall[i]* 100, 2)).rjust(10),
    #                                                                 ))

    # print('IoU:', end=' ')
    # for i in range(len(IoU)):
    #     print('{}%'.format( str(round(IoU[i] * 100, 2))), end=',')
    #     # logging.info('{}%'.format( str(round(IoU[i] * 100, 2))), end=',')
    # print('\n')

    print('F1:', end=' ')
    for i in range(len(IoU)):
        print('{}%'.format(str(round(F1[i] * 100, 2))), end=',')
    print('\n')

    print('precision:', end=' ')
    for i in range(len(IoU)):
        print('{}%'.format(str(round(precision[i] * 100, 2))), end=',')
    print('\n')

    print('recall:', end=' ')
    for i in range(len(IoU)):
        print('{}%'.format(str(round(recall[i] * 100, 2))), end=',')
    print('\n')

    # print('finalresult: mIoU={:.2f}%, acc={:.2f}%, macc={:.2f}%, mF1={:.2f}%,  mprecision={:.2f}%, mrecall={:.2f}%, Kappa={:.2f}'
    #       .format( mIoU * 100, acc * 100, macc * 100, mF1 * 100, mprecision* 100, mrecall * 100, Kappa))
    print(
        'finalresult: acc={:.2f}%, macc={:.2f}%, mF1={:.2f}%, mprecision={:.2f}%, mrecall={:.2f}%, Kappa={:.2f}'
        .format(acc * 100, macc * 100, mF1 * 100, mprecision * 100, mrecall * 100, Kappa))

    # save_log = os.path.join(logpre_path, 'test.log')
    # logging.basicConfig(filename=save_log, level=logging.INFO)