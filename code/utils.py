def get_accuracy(success, failure):
    return success / (success + failure)


def get_total_accuracy(real_success, real_failure, fake_success, fake_failure):
    success = real_success + fake_success
    failure = real_failure + fake_failure
    return success / (success + failure)


def get_metrics(fake_success, fake_failure, real_failure):
    if fake_success == 0 and real_failure == 0:
        return {
            "precision": 0,
            "recall": 0,
            "f1_score": 0
        }


    precision = get_precision(fake_success, real_failure)
    recall = get_recall(fake_success, fake_failure)
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": get_f1_score(precision, recall)
    }


def get_precision(tp, fp):
    return tp / (tp + fp)


def get_recall(tp, fn):
    return tp / (tp + fn)


def get_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)