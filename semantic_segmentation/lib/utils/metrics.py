import numpy as np
from lib.datasets import CityscapesVid

class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()      

class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes,single_iou_class=-1, classes=None):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.single_iou_class = single_iou_class
        self.accs = []
        self.accs_sum = []
        self.ious = []
        self.ious_sum = []
        self.classes = classes

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            ltflatten = lt.flatten()
            lpflatten = lp.flatten()

            conf_mat = self._fast_hist( ltflatten, lpflatten )
            self.confusion_matrix += conf_mat
            iu = np.diag(conf_mat) / (conf_mat.sum(axis=1) + conf_mat.sum(axis=0) - np.diag(conf_mat))
            mean_iu = np.nanmean(iu) if self.single_iou_class < 0 else iu[self.single_iou_class]
            self.ious.append(mean_iu)
            acc = np.diag(conf_mat).sum() / conf_mat.sum()
            self.accs.append(acc)

            hist = self.confusion_matrix
            acc = np.diag(hist).sum() / hist.sum()
            iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
            mean_iu = np.nanmean(iu) if self.single_iou_class < 0 else iu[self.single_iou_class]
            self.ious_sum.append(mean_iu)
            self.accs_sum.append(acc)
            
    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        
        #string+='Class IoU:\n'
        #for k, v in results['Class IoU'].items():
        #    string += "\tclass %d: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self, per_class=False):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu) if self.single_iou_class < 0 else iu[self.single_iou_class]
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

        # if per_class:
        train_id_to_name = [[] for i in range(20)]
        for c in CityscapesVid.classes:
            train_id = c.train_id
            if train_id == 255:
                train_id = 19
            train_id_to_name[train_id].append(c.name)
        train_id_to_name = [', '.join(t) for t in train_id_to_name]
        mapped_classes = train_id_to_name
        cls_iu = dict(zip(mapped_classes, iu))

        if self.classes is not None:
            fine_iu = np.nanmean(iu[self.classes])
        else:
            fine_iu = 0

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
                "Fine mIoU": fine_iu,
            }
        
    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class AverageMeter(object):
    """Computes average values"""
    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()
    
    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0]+=val
            record[1]+=1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]