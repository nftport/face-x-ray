# TODO: this needs to be based on fps actually!


def execute(scores, prediction_threshold=0.5):
    return _percentage_scorer(scores, prediction_threshold)


def _percentage_scorer(scores, prediction_threshold: float):
    binary_outputs = (scores > prediction_threshold)
    score = sum(binary_outputs) / len(binary_outputs)
    return score


consecutive_frames = 30


def _consecutive_frame_scorer(scores, prediction_threshold):
    best_streak = 0
    streak = 0
    for score in scores:
        if score > prediction_threshold:
            streak += 1
        else:
            if streak > best_streak:
                best_streak = streak
            streak = 0

    if best_streak > consecutive_frames:
        return 0.15
    return 0